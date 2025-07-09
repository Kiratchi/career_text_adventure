from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from collections import deque
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import litellm
from litellm import completion
import random

# ==================== CONFIGURATION ====================

# Setup
load_dotenv()
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LLM Configuration
LITELLM_API_KEY = os.getenv('LITELLM_API_KEY')
LITELLM_BASE_URL = os.getenv('LITELLM_BASE_URL', 'https://anast.ita.chalmers.se:4000')

# Configure LiteLLM
litellm.api_base = LITELLM_BASE_URL
litellm.drop_params = True

# Prompt configuration for different LLM calls
PROMPT_CONFIG = {
    "director_analysis": {"max_tokens": 1000, "temperature": 0.7},
    "scenario_creation": {"max_tokens": 1000, "temperature": 0.8},
    "update_history_summary": {"max_tokens": 300, "temperature": 0.6},
    "update_personality_summary": {"max_tokens": 200, "temperature": 0.7},
    "update_current_situation": {"max_tokens": 150, "temperature": 0.6},
    "masters_selection_scenario": {"max_tokens": 500, "temperature": 0.6},
    "simple_scenario": {"max_tokens": 300, "temperature": 0.8},
    "emergency_scenario": {"max_tokens": 200, "temperature": 0.5},
    "chalmers_introduction": {"max_tokens": 400, "temperature": 0.7}  # NEW: Introduction prompt config
}


# ==================== LLM INITIALIZATION ====================

def initialize_llm():
    """Initialize LLM with available models"""
    models_to_try = [
        "claude-haiku-3.5", "claude-sonnet-3.7", "gpt-4.1-2025-04-14"
    ]
    
    try:
        api_key = LITELLM_API_KEY
        if not api_key:
            raise Exception("LITELLM_API_KEY not found in environment")
        
        for model in models_to_try:
            try:
                test_response = litellm.completion(
                    model=model,
                    messages=[{"role": "user", "content": "Hello"}],
                    api_key=api_key,
                    max_tokens=5,
                    base_url=LITELLM_BASE_URL
                )
                logger.info(f"✅ LLM initialized successfully with {model}")
                return model
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}")
                continue
        
        raise Exception("No available models found")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize LLM: {e}")
        return None

# Initialize LLM
llm = initialize_llm()

# ==================== UTILITY FUNCTIONS ====================

def make_llm_call(messages, max_tokens=300, model=None, call_type="general", prompt_name=None):
    """
    Helper function for consistent LLM calls throughout the application.
    """
    if model is None:
        model = llm
    
    if not model:
        raise Exception("No LLM model available")
    
    # Get prompt-specific parameters if prompt_name is provided
    if prompt_name and prompt_name in PROMPT_CONFIG:
        config = PROMPT_CONFIG[prompt_name]
        max_tokens = config.get('max_tokens', max_tokens)
        temperature = config.get('temperature', 0.7)
    else:
        temperature = 0.7
    
    # Extract prompt for logging
    prompt = messages[-1]['content'] if messages else ""
    
    start_time = time.time()
    try:
        response = litellm.completion(
            model=model,
            messages=messages,
            api_key=LITELLM_API_KEY,
            max_tokens=max_tokens,
            temperature=temperature,
            base_url=LITELLM_BASE_URL
        )
        
        raw_response = response.choices[0].message.content.strip()
        processing_time = time.time() - start_time
        
        parsed_response = {"content": raw_response}
        
        # Log the call
        debug_logger.log_llm_call(
            call_type=call_type,
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            raw_response=raw_response,
            parsed_response=parsed_response,
            success=True,
            processing_time=processing_time
        )
        
        return raw_response
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        debug_logger.log_llm_call(
            call_type=call_type,
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            raw_response="",
            parsed_response={},
            success=False,
            error=str(e),
            processing_time=processing_time
        )
        
        logger.error(f"LLM call failed: {e}")
        raise

# Simple prompt loading function
def load_prompt(prompt_name: str, **kwargs) -> str:
    """Load and format a prompt from a text file"""
    try:
        prompt_file = Path("prompts") / f"{prompt_name}.txt"
        
        if not prompt_file.exists():
            logger.error(f"📝 Prompt file not found: {prompt_file}")
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_content = f.read()
        
        logger.info(f"📝 Loaded prompt file: {prompt_name}.txt ({len(prompt_content)} chars)")
        
        # Replace variables
        try:
            formatted_prompt = prompt_content.format(**kwargs)
            logger.info(f"📝 Successfully formatted prompt with {len(kwargs)} variables")
            return formatted_prompt
        except KeyError as e:
            logger.error(f"📝 Missing variable in prompt {prompt_name}: {e}")
            logger.error(f"📝 Available variables: {list(kwargs.keys())}")
            logger.error(f"📝 Prompt content preview: {prompt_content[:200]}...")
            raise KeyError(f"Missing variable in prompt {prompt_name}: {e}")
        except Exception as e:
            logger.error(f"📝 Error formatting prompt {prompt_name}: {e}")
            raise Exception(f"Error formatting prompt {prompt_name}: {e}")
            
    except Exception as e:
        logger.error(f"📝 Error loading prompt {prompt_name}: {e}")
        raise
    
# ==================== INTRODUCTION GENERATOR ====================

def generate_chalmers_introduction(player_name: str) -> str:
    """Generate a personalized introduction to Chalmers for the player"""
    
    if not llm:
        raise Exception("LLM is not available. Please check the server configuration.")
    
    try:
        # Try to load prompt from file
        prompt_content = load_prompt(
            'introduction',
            student_name=player_name
        )
        
        logger.info(f"📝 Loaded introduction prompt for {player_name}")
        
        messages = [{"role": "user", "content": prompt_content}]
        
        introduction = make_llm_call(
            messages,
            call_type="chalmers_introduction",
            prompt_name="chalmers_introduction"
        )
        
        logger.info(f"✅ Generated Chalmers introduction for {player_name}")
        return introduction
        
    except FileNotFoundError as e:
        logger.error(f"Introduction prompt file missing: {e}")
        raise Exception(f"Introduction prompt file not found: prompts/introduction.txt")
    except KeyError as e:
        logger.error(f"Missing variable in introduction prompt: {e}")
        raise Exception(f"Introduction prompt has missing variable: {e}")
    except Exception as e:
        logger.error(f"Error generating introduction: {e}")
        raise Exception(f"Failed to generate introduction: {e}")

# ==================== KNOWLEDGE SYSTEM ====================

class KnowledgeLoader:
    """Load knowledge from external files"""
    
    def __init__(self, knowledge_dir: str = "knowledge"):
        self.knowledge_dir = Path(knowledge_dir)
        self._cache = {}
        self._validate_knowledge_dir()
    
    def _validate_knowledge_dir(self):
        """Validate that knowledge directory exists"""
        if not self.knowledge_dir.exists():
            logger.warning(f"📁 Knowledge directory not found: {self.knowledge_dir}")
        else:
            logger.info(f"📁 Knowledge directory found: {self.knowledge_dir}")
    
    def load_json(self, filename: str) -> Dict[str, Any]:
        """Load JSON knowledge file with caching"""
        if filename in self._cache:
            return self._cache[filename]
        
        file_path = self.knowledge_dir / filename
        if not file_path.exists():
            logger.warning(f"Knowledge file not found: {file_path}")
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._cache[filename] = data
                logger.info(f"📚 Loaded knowledge file: {filename}")
                return data
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return {}
    
    def clear_cache(self):
        """Clear the knowledge cache"""
        self._cache.clear()
        logger.info("🧹 Knowledge cache cleared")

# Global knowledge loader
knowledge = KnowledgeLoader()

# ==================== DATA MODELS ====================

@dataclass
class PlayerProfile:
    name: str = ""
    program: str = ""
    selected_masters_program: str = ""
    personality_description: str = ""
    life_choices: List[str] = field(default_factory=list)
    current_situation: str = "beginning their studies"
    interests: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    year: int = 1
    
    # Enhanced tracking
    comprehensive_history: str = ""
    personality_summary: str = ""
    scenario_types_encountered: Set[str] = field(default_factory=set)
    
    # NEW: Introduction tracking
    has_seen_introduction: bool = False
    introduction_text: str = ""
    
    def add_choice(self, choice_description: str, choice_type: str, choice_data: Dict[str, Any] = None):
        """Add a life choice and update tracking"""
        # Add to legacy format for compatibility
        self.life_choices.append(f"{choice_type}: {choice_description}")
        if len(self.life_choices) > 15:
            self.life_choices = self.life_choices[-15:]
        
        # Track scenario types for avoiding repetition
        self.scenario_types_encountered.add(choice_type)
        
        # Store choice context for AI processing
        choice_context = {
            'choice_description': choice_description,
            'choice_type': choice_type,
            'situation': choice_data.get('situation', '') if choice_data else '',
            'character_implication': choice_data.get('character_implication', '') if choice_data else '',
            'other_options': [opt.get('description', '') for opt in choice_data.get('options', [])] if choice_data else []
        }
        
        return choice_context
    
    def should_advance_year(self) -> bool:
        """Check if student should advance to next year (2 choices per year)"""
        choices_count = len(self.life_choices)
        
        # Year progression: every 2 choices = 1 year
        # Year 1: 0-1 choices
        # Year 2: 2-3 choices  
        # Year 3: 4-5 choices
        # Year 4: 6-7 choices
        # Year 5: 8+ choices
        
        expected_year = min((choices_count // 2) + 1, 5)
        return expected_year > self.year
    
    def advance_year(self):
        """Advance to next year"""
        if self.should_advance_year():
            old_year = self.year
            self.year = min(self.year + 1, 5)
            logger.info(f"📅 Student advanced from Year {old_year} to Year {self.year}")
            return True
        return False
    
    def get_character_summary(self) -> str:
        """Generate character summary for AI prompts"""
        summary_parts = [
            f"{self.name} is a Year {self.year} {self.program} student.",
            f"Current situation: {self.current_situation}"
        ]
        
        if self.selected_masters_program:
            summary_parts.append(f"Selected Master's Program: {self.selected_masters_program}")
        
        if self.personality_summary:
            summary_parts.append(f"Personality: {self.personality_summary}")
        else:
            summary_parts.append(f"Personality: {self.personality_description}")
        
        if self.comprehensive_history:
            summary_parts.append(f"History: {self.comprehensive_history}")
        
        if self.life_choices:
            summary_parts.append("Recent choices:")
            for choice in self.life_choices[-3:]:
                summary_parts.append(f"- {choice}")
        
        return "\n".join(summary_parts)
    
    def has_encountered_scenario_type(self, scenario_type: str) -> bool:
        """Check if student has already encountered this type of scenario"""
        return scenario_type in self.scenario_types_encountered
    
    def get_recent_scenario_types(self, count: int = 3) -> List[str]:
        """Get the types of the most recent scenarios"""
        if not self.life_choices:
            return []
        recent_choices = self.life_choices[-count:]
        return [choice.split(':')[0].strip() for choice in recent_choices if ':' in choice]
    
    def estimate_current_year(self) -> str:
        """Estimate study year based on choices made"""
        return f"Year {self.year}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['scenario_types_encountered'] = list(self.scenario_types_encountered)
        return result

# ==================== DEBUG SYSTEM ====================

@dataclass
class LLMCall:
    """Track individual LLM calls"""
    timestamp: str
    call_type: str  # 'director_analysis', 'scenario_creation', 'profile_update', 'chalmers_introduction', etc.
    prompt: str
    model: str
    max_tokens: int
    raw_response: str
    parsed_response: Dict[str, Any]
    success: bool
    error: Optional[str] = None
    processing_time: float = 0.0

@dataclass
class ToolExecution:
    """Track tool executions"""
    timestamp: str
    tool_name: str
    parameters: Dict[str, Any]
    raw_result: Dict[str, Any]
    success: bool
    error: Optional[str] = None
    processing_time: float = 0.0

@dataclass
class ScenarioGeneration:
    """Track complete scenario generation process"""
    timestamp: str
    session_id: str
    player_profile_snapshot: Dict[str, Any]
    director_analysis: Optional[LLMCall] = None
    tools_executed: List[ToolExecution] = field(default_factory=list)
    scenario_creation: Optional[LLMCall] = None
    final_scenario: Dict[str, Any] = field(default_factory=dict)
    total_processing_time: float = 0.0
    success: bool = False
    error: Optional[str] = None

class AIDebugLogger:
    """Logger for AI decision-making process"""
    
    def __init__(self):
        self.scenario_generations: List[ScenarioGeneration] = []
        self.llm_calls: List[LLMCall] = []
        self.tool_executions: List[ToolExecution] = []
        self.current_scenario_gen: Optional[ScenarioGeneration] = None
    
    def start_scenario_generation(self, session_id: str, player_profile: PlayerProfile):
        """Start tracking a new scenario generation"""
        self.current_scenario_gen = ScenarioGeneration(
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            player_profile_snapshot=self._serialize_profile(player_profile)
        )
        logger.info(f"🐛 DEBUG: Started scenario generation for session {session_id}")
    
    def log_llm_call(self, call_type: str, prompt: str, model: str, max_tokens: int, 
                     raw_response: str, parsed_response: Dict[str, Any], 
                     success: bool, error: Optional[str] = None, processing_time: float = 0.0):
        """Log an LLM call"""
        llm_call = LLMCall(
            timestamp=datetime.now().isoformat(),
            call_type=call_type,
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            raw_response=raw_response,
            parsed_response=parsed_response,
            success=success,
            error=error,
            processing_time=processing_time
        )
        
        self.llm_calls.append(llm_call)
        
        # Add to current scenario generation
        if self.current_scenario_gen:
            if call_type == 'director_analysis':
                self.current_scenario_gen.director_analysis = llm_call
            elif call_type == 'scenario_creation':
                self.current_scenario_gen.scenario_creation = llm_call
        
        logger.info(f"🐛 DEBUG: LLM call logged - {call_type} ({'SUCCESS' if success else 'FAILED'})")
    
    def log_tool_execution(self, tool_name: str, parameters: Dict[str, Any], 
                          raw_result: Dict[str, Any], success: bool, 
                          error: Optional[str] = None, processing_time: float = 0.0):
        """Log a tool execution"""
        tool_exec = ToolExecution(
            timestamp=datetime.now().isoformat(),
            tool_name=tool_name,
            parameters=parameters,
            raw_result=raw_result,
            success=success,
            error=error,
            processing_time=processing_time
        )
        
        self.tool_executions.append(tool_exec)
        
        # Add to current scenario generation
        if self.current_scenario_gen:
            self.current_scenario_gen.tools_executed.append(tool_exec)
        
        logger.info(f"🐛 DEBUG: Tool execution logged - {tool_name} ({'SUCCESS' if success else 'FAILED'})")
    
    def complete_scenario_generation(self, final_scenario: Dict[str, Any], 
                                   success: bool, error: Optional[str] = None, 
                                   total_time: float = 0.0):
        """Complete the current scenario generation"""
        if self.current_scenario_gen:
            self.current_scenario_gen.final_scenario = final_scenario
            self.current_scenario_gen.success = success
            self.current_scenario_gen.error = error
            self.current_scenario_gen.total_processing_time = total_time
            
            self.scenario_generations.append(self.current_scenario_gen)
            logger.info(f"🐛 DEBUG: Scenario generation completed - {'SUCCESS' if success else 'FAILED'}")
            
            self.current_scenario_gen = None
    
    def _serialize_profile(self, profile: PlayerProfile) -> Dict[str, Any]:
        """Serialize profile, handling sets and other non-serializable types"""
        profile_dict = asdict(profile)
        
        # Convert sets to lists for JSON serialization
        if 'scenario_types_encountered' in profile_dict:
            profile_dict['scenario_types_encountered'] = list(profile_dict['scenario_types_encountered'])
        
        return profile_dict
    
    def get_debug_data(self, session_id: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        """Get debug data for API endpoint"""
        # Filter by session if provided
        scenario_gens = self.scenario_generations
        if session_id:
            scenario_gens = [sg for sg in scenario_gens if sg.session_id == session_id]
        
        # Apply limit
        scenario_gens = scenario_gens[-limit:]
        
        return {
            "scenario_generations": [asdict(sg) for sg in scenario_gens],
            "total_llm_calls": len(self.llm_calls),
            "total_tool_executions": len(self.tool_executions),
            "recent_llm_calls": [asdict(call) for call in self.llm_calls[-limit:]],
            "recent_tool_executions": [asdict(tool) for tool in self.tool_executions[-limit:]]
        }

# Global debug logger
debug_logger = AIDebugLogger()

# ==================== PROFILE UPDATE SERVICE ====================

class ProfileUpdateService:
    """Service to update player profile text summaries using AI"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def update_profile_summaries(self, profile: PlayerProfile, choice_context: Dict[str, Any]):
        """Update both history and personality summaries based on new choice"""
        if not self.llm:
            logger.warning("LLM not available for profile updates")
            return
        
        try:
            self._update_history_summary(profile, choice_context)
            self._update_personality_summary(profile, choice_context)
            self._update_current_situation(profile, choice_context)
            
        except Exception as e:
            logger.error(f"Error updating profile summaries: {e}")
    
    def _update_history_summary(self, profile: PlayerProfile, choice_context: Dict[str, Any]):
        """Update the comprehensive history text with better tracking"""
        try:
            # Format other options
            other_options = []
            for opt in choice_context.get('other_options', [])[:3]:
                if isinstance(opt, dict):
                    other_options.append(f"- {opt.get('description', opt)}")
                else:
                    other_options.append(f"- {opt}")
            
            # Load and format the prompt
            prompt_content = load_prompt(
                'update_history_summary',
                student_name=profile.name,
                program=profile.program,
                year=profile.year,
                current_history=profile.comprehensive_history if profile.comprehensive_history else "No history yet - this is their first significant choice.",
                choice_type=choice_context['choice_type'],
                choice_description=choice_context['choice_description'],
                situation=choice_context['situation'],
                character_implication=choice_context['character_implication'],
                other_options='\n'.join(other_options)
            )
            
            if prompt_content:
                messages = [{"role": "user", "content": prompt_content}]
                profile.comprehensive_history = make_llm_call(
                    messages, 
                    call_type="profile_update",
                    prompt_name="update_history_summary"
                )
                logger.info(f"📝 Updated history summary: {profile.comprehensive_history[:100]}...")
            
        except Exception as e:
            logger.error(f"Error updating history summary: {e}")
    
    def _update_personality_summary(self, profile: PlayerProfile, choice_context: Dict[str, Any]):
        """Update the personality summary based on choices made"""
        try:
            recent_choices = '\n'.join([f"- {choice}" for choice in profile.life_choices[-3:]])
            
            prompt_content = load_prompt(
                'update_personality_summary',
                student_name=profile.name,
                current_personality=profile.personality_summary if profile.personality_summary else profile.personality_description,
                choice_description=choice_context['choice_description'],
                character_implication=choice_context['character_implication'],
                recent_choices=recent_choices
            )
            
            if prompt_content:
                messages = [{"role": "user", "content": prompt_content}]
                profile.personality_summary = make_llm_call(
                    messages, 
                    call_type="profile_update",
                    prompt_name="update_personality_summary"
                )
                logger.info(f"📝 Updated personality summary: {profile.personality_summary[:100]}...")
            
        except Exception as e:
            logger.error(f"Error updating personality summary: {e}")
    
    def _update_current_situation(self, profile: PlayerProfile, choice_context: Dict[str, Any]):
        """Update the current situation description"""
        try:
            
            prompt_content = load_prompt(
                'update_current_situation',
                student_name=profile.name,
                year=profile.year,
                program=profile.program,
                choice_description=choice_context['choice_description'],
            )
            
            if prompt_content:
                messages = [{"role": "user", "content": prompt_content}]
                profile.current_situation = make_llm_call(
                    messages, 
                    call_type="profile_update",
                    prompt_name="update_current_situation"
                )
                logger.info(f"📝 Updated current situation: {profile.current_situation}")
            
        except Exception as e:
            logger.error(f"Error updating current situation: {e}")

# ==================== DIRECTOR TOOLS ====================

class DirectorTool(ABC):
    """Abstract base class for director tools"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for the AI"""
        pass
    
    @abstractmethod
    def execute(self, profile: PlayerProfile, **kwargs) -> Dict[str, Any]:
        """Execute the tool"""
        pass

    def execute_with_debug(self, profile: PlayerProfile, **kwargs) -> Dict[str, Any]:
        """Execute tool with debugging"""
        start_time = time.time()
        
        try:
            result = self.execute(profile, **kwargs)
            processing_time = time.time() - start_time
            
            debug_logger.log_tool_execution(
                tool_name=self.name,
                parameters=kwargs,
                raw_result=result,
                success=True,
                processing_time=processing_time
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            debug_logger.log_tool_execution(
                tool_name=self.name,
                parameters=kwargs,
                raw_result={},
                success=False,
                error=str(e),
                processing_time=processing_time
            )
            
            raise

class GetMastersProgramsTool(DirectorTool):
    """Tool to get available master's programs for the student's bachelor program"""

    @property
    def name(self) -> str:
        return "get_masters_programs"

    @property
    def description(self) -> str:
        return "Get available master's programs for the student's bachelor program."

    def execute(self, profile: PlayerProfile, **kwargs) -> Dict[str, Any]:
        masters_data = knowledge.load_json('masters_programs.json')
        all_programs = masters_data.get('masters_programs', [])

        # Filter programs that are eligible for the student's bachelor program
        available_programs = [
            p for p in all_programs if profile.program in p.get('eligible_programs', [])
        ]

        # Optional difficulty filter
        difficulty = kwargs.get('difficulty_level')
        if difficulty:
            available_programs = [
                p for p in available_programs if p.get('difficulty', 'medium') == difficulty
            ]

        recommendations = self._generate_recommendations(profile, available_programs)

        return {
            "bachelor_program": profile.program,
            "available_masters": available_programs,
            "total_programs": len(available_programs),
            "recommendations": recommendations,
            "student_year": profile.year,
            "ready_for_selection": profile.year >= 3
        }

    def _generate_recommendations(self, profile: PlayerProfile, programs: List[Dict]) -> List[Dict]:
        choices_text = ' '.join(profile.life_choices).lower()
        recommendations = []

        for program in programs:
            match_score = 0
            reasons = []
            program_name = program.get('name', '').lower()

            for specialization in program.get('specializations', []):
                if any(word in choices_text for word in specialization.lower().split() if len(word) > 3):
                    match_score += 2
                    reasons.append(f"Interest in {specialization.lower()}")
                    break

            for career in program.get('career_paths', []):
                if any(word in choices_text for word in career.lower().split() if len(word) > 3):
                    match_score += 1
                    reasons.append(f"Aligns with {career.lower()} path")
                    break

            keyword_hits = sum(
                1 for kw in ['robotics', 'software', 'ai', 'energy', 'entrepreneurship']
                if kw in program_name and kw in choices_text
            )
            if keyword_hits:
                match_score += 3
                reasons.append("Relevant to your previous experiences")

            # Academic focus
            academic_signals = sum(
                1 for word in ['academic', 'study', 'research', 'grade']
                if any(word in choice.lower() for choice in profile.life_choices)
            )
            if program.get('difficulty') == 'high' and academic_signals >= 2:
                match_score += 1
                reasons.append("Good for strong academic profile")

            # Practical focus
            if program.get('difficulty') == 'low' and academic_signals == 0:
                match_score += 1
                reasons.append("Suited for hands-on focus")

            # Leadership/industry experience
            leadership = sum(
                1 for word in ['leadership', 'team', 'organize', 'manage', 'social']
                if any(word in choice.lower() for choice in profile.life_choices)
            )
            if leadership >= 2 and any(k in program_name for k in ['management', 'entrepreneurship']):
                match_score += 2
                reasons.append("Fits your leadership background")

            if match_score > 0:
                recommendations.append({
                    "program": program,
                    "match_score": match_score,
                    "reasons": list(set(reasons)),
                    "recommendation_level": "high" if match_score >= 5 else "medium" if match_score >= 3 else "low"
                })

        return sorted(recommendations, key=lambda x: x['match_score'], reverse=True)[:4]



class GetCoursesTool(DirectorTool):
    """Tool to get available courses for a program"""
    
    @property
    def name(self) -> str:
        return "get_courses"
    
    @property
    def description(self) -> str:
        return "Get available courses for the student's program. Use this when creating course selection scenarios."
    
    def execute(self, profile: PlayerProfile, **kwargs) -> Dict[str, Any]:
        courses = knowledge.load_json('chalmers_courses.json')
        program_courses = courses.get(profile.program, {})
        
        # Get courses for current year
        year_key = f'year_{profile.year}'
        year_courses = program_courses.get(year_key, {})
        
        return {
            "core_courses": year_courses.get('core', []),
            "electives": year_courses.get('electives', []),
            "program": profile.program,
            "year": profile.year
        }

class GetOrganizationsTool(DirectorTool):
    """Tool to get student organizations"""
    
    @property
    def name(self) -> str:
        return "get_organizations"
    
    @property
    def description(self) -> str:
        return "Get student organizations and clubs. Use this for social opportunities and extracurricular activities."
    
    def execute(self, profile: PlayerProfile, **kwargs) -> Dict[str, Any]:
        orgs = knowledge.load_json('student_organizations.json')
        
        # Filter by interest if specified
        interest_filter = kwargs.get('interest_category')
        if interest_filter and interest_filter in orgs:
            return {interest_filter: orgs[interest_filter]}
        
        return orgs

class GetCompaniesTool(DirectorTool):
    """Tool to get companies for internships and career opportunities"""
    
    @property
    def name(self) -> str:
        return "get_companies"
    
    @property
    def description(self) -> str:
        return "Get companies offering internships and career opportunities. Use for career preparation and summer plans."
    
    def _get_program_category(self, program_name: str) -> str:
        """Categorize programs for better scenario matching using JSON data"""
        try:
            programs_data = knowledge.load_json('programs.json')
            programs = programs_data.get('chalmers_engineering_programs', [])
            
            for program in programs:
                if program.get('name') == program_name:
                    field = program.get('field', '')
                    if any(keyword in field.lower() for keyword in ['computer', 'information', 'automation', 'electrical']):
                        return 'tech'
                    elif any(keyword in field.lower() for keyword in ['mechanical', 'civil', 'chemical']):
                        return 'engineering'
                    elif any(keyword in field.lower() for keyword in ['architecture', 'design']):
                        return 'design'
                    elif any(keyword in field.lower() for keyword in ['physics', 'mathematics', 'global']):
                        return 'science'
                    elif any(keyword in field.lower() for keyword in ['biotechnology', 'medical']):
                        return 'bio'
                    else:
                        return 'general'
            return 'general'
        except Exception as e:
            logger.warning(f"Error loading programs.json for categorization: {e}")
            return 'general'
    
    def execute(self, profile: PlayerProfile, **kwargs) -> Dict[str, Any]:
        companies = knowledge.load_json('gothenburg_companies.json')
        
        # Enhanced filtering by program relevance AND category
        program_category = self._get_program_category(profile.program)
        
        relevant_companies = {}
        for industry, company_list in companies.items():
            relevant_companies[industry] = []
            for company in company_list:
                relevant_programs = company.get('relevant_programs', [])
                relevant_categories = company.get('relevant_categories', [])
                
                if (profile.program in relevant_programs or 
                    program_category in relevant_categories or
                    'All programs' in relevant_programs):
                    relevant_companies[industry].append(company)
        
        return relevant_companies

class GetProgramsExplanation:
    """Helper to extract program explanations from programs.json"""

    @staticmethod
    def get_explanation(program_name: str) -> str:
        try:
            programs_data = knowledge.load_json('programs.json')
            programs = programs_data.get('chalmers_engineering_programs', [])
            
            for program in programs:
                if program.get('name') == program_name:
                    return program.get('explanation', 'No explanation available.')
            
            return 'Program explanation not found.'
        except Exception as e:
            logger.warning(f"Error fetching program explanation for {program_name}: {e}")
            return 'Program explanation not available.'
        

class GetUniversitySystemTool(DirectorTool):
    """Tool to get Swedish university system information"""
    
    @property
    def name(self) -> str:
        return "get_university_system"
    
    @property
    def description(self) -> str:
        return "Get information about Swedish university system, grading scale, academic calendar, degree requirements, and university culture. Use when creating scenarios about academic progression, grades, or university procedures."
    
    def execute(self, profile: PlayerProfile, **kwargs) -> Dict[str, Any]:
        system_info = knowledge.load_json('swedish_university_info.json')
        
        # Filter by topic if specified
        topic = kwargs.get('topic')
        if topic and topic in system_info:
            return {topic: system_info[topic]}
        
        return system_info

class GetDegreeRequirementsTool(DirectorTool):
    """Tool to check degree requirements for student's program"""
    
    @property
    def name(self) -> str:
        return "get_degree_requirements"
    
    @property
    def description(self) -> str:
        return "Check degree requirements, credit requirements, mandatory courses, and graduation criteria for the student's program. Use when creating scenarios about course planning or academic milestones."
    
    def execute(self, profile: PlayerProfile, **kwargs) -> Dict[str, Any]:
        # Get program-specific requirements
        courses = knowledge.load_json('chalmers_courses.json')
        program_info = courses.get(profile.program, {})
        
        # Calculate completion status
        total_courses_taken = len([choice for choice in profile.life_choices if 'course' in choice.lower()])
        
        requirements = {
            "program": profile.program,
            "current_year": profile.year,
            "total_years": 5,  # Master's program
            "courses_taken": total_courses_taken,
            "graduation_requirements": {
                "bachelor_credits": 180,
                "master_credits": 120,
                "thesis_required": True,
                "internship_recommended": True
            }
        }
        
        # Add specific program requirements if available
        if program_info:
            requirements["program_structure"] = program_info
        
        return requirements

class GetAcademicCalendarTool(DirectorTool):
    """Tool to get academic calendar information"""
    
    @property
    def name(self) -> str:
        return "get_academic_calendar"
    
    @property
    def description(self) -> str:
        return "Get academic calendar information including exam periods, registration deadlines, holidays, and semester structure. Use for time-sensitive scenarios and academic planning."
    
    def execute(self, profile: PlayerProfile, **kwargs) -> Dict[str, Any]:
        calendar = knowledge.load_json('academic_calendar.json')
        
        # If no file exists, provide typical Swedish academic calendar
        if not calendar:
            calendar = {
                "current_period": "Autumn semester",
                "upcoming_deadlines": [
                    {"event": "Course registration", "date": "Early August", "importance": "high"},
                    {"event": "First exam period", "date": "October", "importance": "medium"},
                    {"event": "Winter break", "date": "December-January", "importance": "low"},
                    {"event": "Final exams", "date": "January", "importance": "high"}
                ],
                "semester_structure": {
                    "autumn": "September - January",
                    "spring": "January - June",
                    "summer": "June - August (optional courses)"
                },
                "important_periods": [
                    "Registration: July-August",
                    "Midterm exams: October/March", 
                    "Final exams: January/May-June",
                    "Thesis deadlines: May/October"
                ]
            }
        
        return calendar



class GetCampusFacilitiesTool(DirectorTool):
    """Tool to get campus facilities information"""
    
    @property
    def name(self) -> str:
        return "get_campus_facilities"
    
    @property
    def description(self) -> str:
        return "Get campus facilities like libraries, labs, study spaces, sports facilities, restaurants. Use for scenarios about where to study, work on projects, or spend time on campus."
    
    def execute(self, profile: PlayerProfile, **kwargs) -> Dict[str, Any]:
        facilities = knowledge.load_json('campus_facilities.json')
        
        # If no file exists, provide basic facilities
        if not facilities:
            facilities = {
                "libraries": [
                    {"name": "Architecture and Civil Engineering Library", "specialty": "Engineering resources"},
                    {"name": "Math and Science Library", "specialty": "Technical literature"},
                    {"name": "Central Library", "specialty": "General collection"}
                ],
                "labs": [
                    {"name": "Computer Labs", "equipment": "High-end workstations, specialized software"},
                    {"name": "Engineering Labs", "equipment": "Testing equipment, prototyping tools"},
                    {"name": "Project Rooms", "equipment": "Collaborative spaces, whiteboards"}
                ],
                "study_spaces": [
                    {"name": "Group Study Rooms", "capacity": "4-8 people"},
                    {"name": "Silent Study Areas", "capacity": "Individual work"},
                    {"name": "Collaborative Spaces", "capacity": "Flexible seating"}
                ],
                "other": [
                    {"name": "Student Union Building", "services": "Food, social activities"},
                    {"name": "Sports Center", "facilities": "Gym, courts, pool"},
                    {"name": "Maker Space", "equipment": "3D printers, electronics lab"}
                ]
            }
        
        # Filter by facility type if specified
        facility_type = kwargs.get('facility_type')
        if facility_type and facility_type in facilities:
            return {facility_type: facilities[facility_type]}
        
        return facilities

class GetStudyAbroadTool(DirectorTool):
    """Tool to get study abroad opportunities"""
    
    @property
    def name(self) -> str:
        return "get_study_abroad"
    
    @property
    def description(self) -> str:
        return "Get study abroad opportunities, exchange programs, and international experiences. Use when student is ready for global opportunities."
    
    def execute(self, profile: PlayerProfile, **kwargs) -> Dict[str, Any]:
        abroad_programs = knowledge.load_json('study_abroad_programs.json')
        
        # If no file exists, provide typical programs
        if not abroad_programs:
            abroad_programs = {
                "erasmus_programs": [
                    {"university": "TU Delft", "country": "Netherlands", "specialty": "Engineering", "duration": "1 semester"},
                    {"university": "ETH Zurich", "country": "Switzerland", "specialty": "Technology", "duration": "1 year"},
                    {"university": "Technical University of Munich", "country": "Germany", "specialty": "Engineering", "duration": "1 semester"}
                ],
                "global_programs": [
                    {"university": "MIT", "country": "USA", "specialty": "Technology", "duration": "Summer program"},
                    {"university": "University of Tokyo", "country": "Japan", "specialty": "Robotics", "duration": "1 semester"},
                    {"university": "National University of Singapore", "country": "Singapore", "specialty": "Engineering", "duration": "1 semester"}
                ],
                "requirements": {
                    "minimum_year": 2,
                    "gpa_requirement": "Good academic standing",
                    "language_requirements": "English proficiency"
                }
            }
        
        # Filter by region if specified
        region = kwargs.get('region')
        if region:
            filtered_programs = {}
            for program_type, programs in abroad_programs.items():
                if isinstance(programs, list):
                    filtered = [p for p in programs if region.lower() in p.get('country', '').lower()]
                    if filtered:
                        filtered_programs[program_type] = filtered
            return filtered_programs if filtered_programs else abroad_programs
        
        return abroad_programs

class CheckPrerequisitesTool(DirectorTool):
    """Tool to check prerequisites for courses or opportunities"""
    
    @property
    def name(self) -> str:
        return "check_prerequisites"
    
    @property
    def description(self) -> str:
        return "Check if student has completed prerequisites for advanced courses, programs, or opportunities. Use to determine what options are realistically available."
    
    def execute(self, profile: PlayerProfile, **kwargs) -> Dict[str, Any]:
        # Analyze student's background
        total_choices = len(profile.life_choices)
        has_technical_focus = any('technical' in choice.lower() or 'programming' in choice.lower() for choice in profile.life_choices)
        has_social_experience = any('social' in choice.lower() or 'club' in choice.lower() or 'organization' in choice.lower() for choice in profile.life_choices)
        has_industry_experience = any('internship' in choice.lower() or 'company' in choice.lower() for choice in profile.life_choices)
        
        prerequisites = {
            "year": profile.year,
            "total_experiences": total_choices,
            "qualifications": {
                "advanced_courses": profile.year >= 2,
                "thesis_projects": profile.year >= 3,
                "study_abroad": profile.year >= 2 and total_choices >= 3,
                "leadership_roles": has_social_experience,
                "research_opportunities": has_technical_focus and profile.year >= 2,
                "industry_internships": profile.year >= 2,
                "master_programs": profile.year >= 4,
                "phd_programs": profile.year >= 4 and has_technical_focus
            },
            "recommendations": []
        }
        
        # Add specific recommendations
        if not has_social_experience:
            prerequisites["recommendations"].append("Consider joining student organizations to build social skills")
        if not has_technical_focus and profile.program in ["Computer Science and Engineering", "Electrical Engineering"]:
            prerequisites["recommendations"].append("Focus on technical skill development")
        if profile.year >= 3 and not has_industry_experience:
            prerequisites["recommendations"].append("Seek internship opportunities for industry experience")
        
        return prerequisites


# ==================== GAME DIRECTOR ====================

class GameDirector:
    """AI Director that decides what happens next in the game"""
    
    def __init__(self, llm):
        self.llm = llm
        self.tools = {
            # Core university data tools
            'get_courses': GetCoursesTool(),
            'get_organizations': GetOrganizationsTool(),
            'get_companies': GetCompaniesTool(),
            'get_university_system': GetUniversitySystemTool(),
            'get_degree_requirements': GetDegreeRequirementsTool(),
            'get_campus_facilities': GetCampusFacilitiesTool(),
            'get_academic_calendar': GetAcademicCalendarTool(),
            'get_study_abroad': GetStudyAbroadTool(),
            
            # Master's program selection tools
            'get_masters_programs': GetMastersProgramsTool(),
            
            # Analysis and decision support tools
            'check_prerequisites': CheckPrerequisitesTool(),
        }

    def _get_tool_descriptions(self) -> str:
        """Get descriptions of all available tools"""
        tool_descriptions = []
        for tool_name, tool in self.tools.items():
            tool_descriptions.append(f"- {tool_name}: {tool.description}")
        return "\n".join(tool_descriptions)
    
    def _execute_tool(self, tool_name: str, profile: PlayerProfile, **kwargs) -> Dict[str, Any]:
        """Execute a specific tool with debugging"""
        if tool_name not in self.tools:
            logger.warning(f"🔧 Tool '{tool_name}' not found")
            return {"error": f"Tool '{tool_name}' not found"}
        
        try:
            logger.info(f"🔧 TOOL EXECUTION: {tool_name}")
            if kwargs:
                logger.info(f"   Parameters: {kwargs}")
            
            # Use debug-enhanced execution
            result = self.tools[tool_name].execute_with_debug(profile, **kwargs)
            
            # Log summary of results
            if 'error' not in result:
                if tool_name == 'get_courses':
                    electives = result.get('electives', [])
                    logger.info(f"   → Found {len(electives)} elective courses for {result.get('program', 'unknown')} Year {result.get('year', 'unknown')}")
                elif tool_name == 'get_organizations':
                    total_orgs = sum(len(orgs) for orgs in result.values() if isinstance(orgs, list))
                    categories = list(result.keys())
                    logger.info(f"   → Found {total_orgs} organizations across {len(categories)} categories: {', '.join(categories)}")
                elif tool_name == 'get_companies':
                    total_companies = sum(len(companies) for companies in result.values() if isinstance(companies, list))
                    industries = list(result.keys())
                    logger.info(f"   → Found {total_companies} companies across {len(industries)} industries: {', '.join(industries)}")
                elif tool_name == 'get_masters_programs':
                    available = result.get('available_masters', [])
                    recommendations = result.get('recommendations', [])
                    logger.info(f"   → Found {len(available)} available master's programs with {len(recommendations)} recommendations")
                elif tool_name == 'get_history_context':
                    activities = result.get('activities_already_done', [])
                    warnings = result.get('repetition_warnings', [])
                    logger.info(f"   → Student has done {len(activities)} activities with {len(warnings)} repetition warnings")
                elif tool_name == 'analyze_progress':
                    choices = result.get('total_choices', 0)
                    missing = result.get('missing_areas', [])
                    logger.info(f"   → Student has {choices} choices, missing areas: {', '.join(missing)}")
            else:
                logger.error(f"   → Tool error: {result['error']}")
            
            return result
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"error": str(e)}

    def _make_llm_call_with_retry(self, messages: List[Dict], call_type: str, prompt_name: str, max_retries: int = 3) -> str:
        """Make LLM call with retry logic"""
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🤖 LLM CALL ATTEMPT {attempt + 1}/{max_retries} ({call_type})")
                
                response = make_llm_call(messages, call_type=call_type, prompt_name=prompt_name)
                
                # Basic validation - check if response is not empty and seems reasonable
                if not response or len(response.strip()) < 10:
                    raise ValueError(f"LLM response too short or empty: {len(response)} characters")
                
                logger.info(f"🤖 LLM CALL SUCCESS on attempt {attempt + 1}")
                return response
                
            except Exception as e:
                logger.warning(f"🤖 LLM CALL FAILED (attempt {attempt + 1}): {e}")
                
                # If this was the last attempt, we'll fall through to return None
                if attempt == max_retries - 1:
                    logger.error(f"🤖 LLM CALL FAILED after {max_retries} attempts")
                    break
                
                # Wait a bit before retrying (exponential backoff)
                wait_time = 0.5 * (2 ** attempt)
                logger.info(f"🤖 Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        
        return None

    def _create_ai_crash_fallback(self, profile: PlayerProfile) -> Dict[str, Any]:
        """Create a humorous fallback when the AI has completely crashed"""
        
        crash_situations = [
            f"The AI Director has blue-screened while trying to figure out what {profile.name} should do next. In the confusion, three random choices have appeared:",
            f"ERROR 404: Plot not found. The AI Director is currently stuck in an infinite loop thinking about {profile.name}'s future. Please choose from these emergency backup options:",
            f"The AI Director has gone for a coffee break after encountering a 'critical thinking error'. While we wait for it to return, here are some placeholder choices:",
            f"SYSTEM MALFUNCTION: The AI Director tried to process {profile.name}'s complex personality and promptly gave up. These generic choices have been auto-generated instead:",
            f"The AI Director is experiencing technical difficulties (probably tried to divide by zero while calculating {profile.name}'s future). Emergency choices activated:",
            f"🚨 EMERGENCY PROTOCOL ACTIVATED 🚨 The AI Director has crashed spectacularly while analyzing {profile.name}'s situation. Manual override engaged:",
            f"The AI Director attempted to understand {profile.name}'s life choices and immediately filed for early retirement. Backup choices loading..."
        ]
        
        crash_options = [
            {
                "id": "choice_1_crash",
                "description": "Choose Option 1 (The AI doesn't know what this does)",
                "character_implication": "somehow affects your character in mysterious ways"
            },
            {
                "id": "choice_2_crash", 
                "description": "Choose Option 2 (This might be related to your studies? Maybe?)",
                "character_implication": "does something vaguely academic-sounding"
            },
            {
                "id": "choice_3_crash",
                "description": "Choose Option 3 (The AI is just guessing at this point)",
                "character_implication": "results in unpredictable character development"
            },
            {
                "id": "reboot_ai",
                "description": "Try to reboot the AI Director (50% chance of success, 50% chance of making things worse)",
                "character_implication": "either fixes the AI or causes even more chaos"
            }
        ]
        
        # Pick a random crash situation
        situation = random.choice(crash_situations)
        
        return {
            "type": "ai_crash",
            "title": "🤖 AI DIRECTOR MALFUNCTION 🤖",
            "situation": situation,
            "options": crash_options
        }

    def should_trigger_masters_selection(self, profile: PlayerProfile) -> bool:
        """Check if we should trigger master's program selection when transitioning 3->4"""
        # Only trigger when student is exactly in year 3 and about to advance to year 4
        if profile.year != 3:
            return False
        
        # Don't trigger if already selected
        if profile.selected_masters_program:
            return False
        
        # Don't trigger if already seen master's selection
        if any('master' in choice.lower() for choice in profile.life_choices):
            return False
        
        # Check if student is about to advance to year 4
        choices_count = len(profile.life_choices)
        # If they have 5 choices (end of year 3 with 2 choices per year), they should select master's before year 4
        return choices_count >= 5

    def create_masters_selection_scenario(self, profile: PlayerProfile) -> Dict[str, Any]:
        """Create a master's program selection scenario"""
        
        # Get available programs using the tool
        programs_data = self._execute_tool('get_masters_programs', profile)
        
        available_programs = programs_data.get('available_masters', [])
        recommendations = programs_data.get('recommendations', [])
        
        if not available_programs:
            return self._create_ai_crash_fallback(profile)
        
        # Create urgency text
        urgency_text = "It's time to choose your master's program!" if profile.year == 4 else "You need to select your master's specialization before advancing to Year 4."
        
        # Use simple prompt loading
        situation = load_prompt(
            'masters_selection_scenario',
            student_name=profile.name,
            year=profile.year,
            program=profile.program,
            urgency_text=urgency_text,
            available_programs_count=len(available_programs),
            recommendations_count=len(recommendations)
        )
        
        # Create options from available programs (limit to 4 for UI)
        options = []
        
        # Include top recommendations first
        recommended_programs = [rec['program'] for rec in recommendations[:2]]
        other_programs = [p for p in available_programs if p not in recommended_programs]
        
        # Take top 2 recommendations + 2 others
        selected_programs = recommended_programs + other_programs[:2]
        
        for i, program in enumerate(selected_programs[:4]):
            option_text = f"Choose {program['name']} ({program['code']}) - {program['description']}"
            
            # Add recommendation note if it's recommended
            if program in recommended_programs:
                option_text += f" [Recommended based on your background]"
            
            options.append({
                "id": f"masters_{program['code'].lower()}",
                "description": option_text,
                "character_implication": f"Specializes in {program['name']}, developing expertise in {', '.join(program['specializations'][:2])} and preparing for careers in {', '.join(program['career_paths'][:2])}"
            })
        
        return {
            "type": "masters_selection",
            "title": f"Master's Program Selection - Year {profile.year}",
            "situation": situation,
            "options": options,
            "metadata": {
                "all_programs": available_programs,
                "recommendations": recommendations,
                "selection_year": profile.year
            }
        }

    def decide_next_scenario(self, profile: PlayerProfile) -> Dict[str, Any]:
        """Simplified scenario decision using pre-processed profile summaries"""
        
        # Add recursion prevention
        if hasattr(self, '_generating_scenario') and self._generating_scenario:
            logger.warning("🎬 RECURSION DETECTED: Already generating scenario, returning fallback")
            return self._create_ai_crash_fallback(profile)
        
        self._generating_scenario = True
        
        try:
            # Get the session_id from the request context or use default
            session_id = getattr(request, 'json', {}).get('session_id', 'default') if request else 'default'
            
            # Start debug tracking
            debug_logger.start_scenario_generation(session_id, profile)
            
            logger.info(f"🎬 DIRECTOR STARTING: Decision for {profile.name} ({profile.program}, Year {profile.year})")
            logger.info(f"🐛 DEBUG: Using session_id: {session_id}")
            
            # Check if student should advance year first
            year_advanced = profile.advance_year()
            if year_advanced:
                logger.info(f"📅 Student advanced to Year {profile.year}")
            
            # Check if we should trigger master's selection
            if self.should_trigger_masters_selection(profile):
                logger.info("🎓 TRIGGERING MASTER'S PROGRAM SELECTION (Year 3 -> Year 4 transition)")
                scenario = self.create_masters_selection_scenario(profile)
                debug_logger.complete_scenario_generation(scenario, True, total_time=0.0)
                return scenario
            
            logger.info(f"   Current situation: {profile.current_situation}")
            logger.info(f"   Personality: {profile.personality_description}")
            logger.info(f"   Choices made so far: {len(profile.life_choices)}")
            
            # Log recent choices for context
            if profile.life_choices:
                logger.info("   Recent choices:")
                for choice in profile.life_choices[-3:]:
                    logger.info(f"     - {choice}")
            
            if not self.llm:
                logger.warning("🎬 DIRECTOR: LLM not available, using fallback")
                fallback_scenario = self._create_ai_crash_fallback(profile)
                debug_logger.complete_scenario_generation(fallback_scenario, False, error="LLM not available")
                return fallback_scenario

            start_time = time.time()
            
            # PHASE 1: Load director analysis prompt using pre-processed summaries
            logger.info("🎬 PHASE 1: Loading director analysis prompt with profile summaries...")
            prompt_content = load_prompt(
                'director_analysis',
                student_name=profile.name,
                program=profile.program,
                year=profile.year,
                program_explanation=GetProgramsExplanation.get_explanation(profile.program),
                comprehensive_history=profile.comprehensive_history or "This is their first significant decision at university.",
                personality_summary=profile.personality_summary or profile.personality_description,
                tool_descriptions=self._get_tool_descriptions()
            )
            
            if not prompt_content:
                logger.error("🎬 PHASE 1: Director analysis prompt failed to load!")
                fallback_scenario = self._create_ai_crash_fallback(profile)
                debug_logger.complete_scenario_generation(fallback_scenario, False, error="Director analysis prompt failed")
                return fallback_scenario
            
            logger.info(f"🎬 PHASE 1: Prompt loaded successfully ({len(prompt_content)} chars)")
            
            # PHASE 2: Get AI analysis with retry
            logger.info("🎬 PHASE 2: Getting AI analysis with retry logic...")
            messages = [{"role": "user", "content": prompt_content}]
            response = self._make_llm_call_with_retry(
                messages, 
                call_type="director_analysis",
                prompt_name="director_analysis"
            )
            
            # Check if LLM call failed after all retries
            if response is None:
                logger.error("🎬 PHASE 2: AI analysis failed after all retries")
                fallback_scenario = self._create_ai_crash_fallback(profile)
                debug_logger.complete_scenario_generation(fallback_scenario, False, error="AI analysis failed after retries")
                return fallback_scenario
            
            logger.info(f"🎬 PHASE 2: AI analysis received ({len(response)} chars)")
            
            # Parse the response
            analysis = self._parse_director_response(response)
            
            logger.info("🎬 DIRECTOR ANALYSIS:")
            logger.info(f"   Reasoning: {analysis.get('analysis', 'No analysis provided')[:100]}...")
            logger.info(f"   Scenario Type: {analysis.get('scenario_type', 'unknown')}")
            logger.info(f"   Tools Planned: {len(analysis.get('tools_to_use', []))} tools")
            
            # PHASE 3: Execute the tools the AI requested
            logger.info("🎬 PHASE 3: Executing requested tools...")
            tool_results = {}
            
            for i, tool_call in enumerate(analysis.get('tools_to_use', []), 1):
                logger.info(f"🎬 TOOL {i}/{len(analysis.get('tools_to_use', []))}: {tool_call}")
                tool_name, params = self._parse_tool_call(tool_call)
                if tool_name:
                    tool_results[tool_name] = self._execute_tool(tool_name, profile, **params)
                else:
                    logger.warning(f"   → Could not parse tool call: {tool_call}")
            
            logger.info(f"🎬 PHASE 4: Creating scenario with {len(tool_results)} tool results...")
            
            # PHASE 4: Create the actual scenario with retry
            scenario = self._create_scenario_with_context_retry(
                profile, 
                analysis.get('scenario_type', 'general'),
                analysis.get('analysis', ''),
                tool_results
            )
            
            total_time = time.time() - start_time
            
            # Complete debug tracking
            debug_logger.complete_scenario_generation(
                final_scenario=scenario,
                success=True,
                total_time=total_time
            )
            
            logger.info(f"🎬 DIRECTOR COMPLETE: Generated scenario '{scenario.get('title', 'Unknown')}' with {len(scenario.get('options', []))} options")
            
            return scenario
            
        except Exception as e:
            total_time = time.time() - start_time if 'start_time' in locals() else 0.0
            
            debug_logger.complete_scenario_generation(
                final_scenario={},
                success=False,
                error=str(e),
                total_time=total_time
            )
            
            logger.error(f"🎬 DIRECTOR ERROR: {e}")
            import traceback
            logger.error(f"🎬 DIRECTOR TRACEBACK: {traceback.format_exc()}")
            return self._create_ai_crash_fallback(profile)
        
        finally:
            # Always clear the recursion flag
            self._generating_scenario = False

    def _create_scenario_with_context_retry(self, profile: PlayerProfile, scenario_type: str, analysis: str, tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create the final scenario using AI with context from tools and history awareness WITH RETRY"""
        
        logger.info("🎬 PHASE 5: Creating final scenario with AI and retry logic...")
        logger.info(f"   Scenario type: {scenario_type}")
        logger.info(f"   Using context from {len(tool_results)} tools")
        
        context_summary = self._summarize_tool_results(tool_results)
        
        # Get history context for repetition avoidance
        history_context = tool_results.get('history_context', {})
        
        try:
            # Use simple prompt loading for scenario creation
            prompt_content = load_prompt(
                'scenario_creation',
                student_name=profile.name,
                program=profile.program,
                year=profile.year,
                program_explanation=GetProgramsExplanation.get_explanation(profile.program),
                comprehensive_history=profile.comprehensive_history or "This is their first significant decision at university.",
                personality_summary=profile.personality_summary or profile.personality_description,
                analysis=analysis,
                scenario_type=scenario_type,
                activities_done=', '.join(history_context.get('activities_already_done', [])),
                organizations_joined=', '.join(history_context.get('organizations_joined', [])),
                courses_taken=', '.join(history_context.get('courses_taken', [])),
                companies_applied=', '.join(history_context.get('companies_applied', [])),
                repetition_warnings='\n'.join(history_context.get('repetition_warnings', ['No repetition concerns'])),
                variety_suggestions='\n'.join(history_context.get('variety_suggestions', ['Continue exploring new areas'])),
                context_summary=context_summary
            )
            
            # ADD DEBUG LOGGING
            logger.info(f"🎬 PROMPT LOADED: {len(prompt_content)} characters")
            if not prompt_content:
                logger.error("🎬 ERROR: Prompt content is empty!")
                return self._create_ai_crash_fallback(profile)
            
            # Log first 200 chars of prompt for debugging
            logger.info(f"🎬 PROMPT PREVIEW: {prompt_content[:200]}...")
            
            # Make LLM call with retry
            messages = [{"role": "user", "content": prompt_content}]
            response = self._make_llm_call_with_retry(
                messages, 
                call_type="scenario_creation",
                prompt_name="scenario_creation"
            )
            
            # Check if LLM call failed after all retries
            if response is None:
                logger.error("🎬 SCENARIO CREATION: AI failed after all retries")
                return self._create_ai_crash_fallback(profile)
            
            # ADD DEBUG LOGGING
            logger.info(f"🎬 AI RESPONSE: {len(response)} characters")
            logger.info(f"🎬 RESPONSE PREVIEW: {response[:300]}...")
            
            # Parse JSON response
            scenario = self._parse_scenario_json(response)
            
            # ADD DEBUG LOGGING
            if scenario.get('options'):
                logger.info(f"🎬 SCENARIO PARSED SUCCESSFULLY: {len(scenario['options'])} options")
            else:
                logger.error("🎬 ERROR: Scenario parsing failed or no options found")
                logger.error(f"🎬 PARSED SCENARIO: {scenario}")
                return self._create_ai_crash_fallback(profile)
            
            logger.info("🎬 SCENARIO GENERATED:")
            logger.info(f"   Title: {scenario.get('title', 'Unknown')}")
            logger.info(f"   Options: {len(scenario.get('options', []))} choices available")
            
            return scenario
            
        except Exception as e:
            logger.error(f"🎬 Error creating scenario: {e}")
            logger.error(f"🎬 Exception type: {type(e)}")
            import traceback
            logger.error(f"🎬 Full traceback: {traceback.format_exc()}")
            return self._create_ai_crash_fallback(profile)

    def _summarize_tool_results(self, tool_results: Dict[str, Any]) -> str:
        """Summarize tool results for scenario prompt context"""
        summary_parts = []

        # History context first
        history = tool_results.get('history_context', {})
        if history:
            if history.get('activities_already_done'):
                summary_parts.append("PREVIOUS ACTIVITIES:")
                for activity in history['activities_already_done']:
                    summary_parts.append(f"- {activity}")

            if history.get('variety_suggestions'):
                summary_parts.append("SUGGESTED NEW AREAS:")
                for suggestion in history['variety_suggestions']:
                    summary_parts.append(f"- {suggestion}")

            summary_parts.append("")  # spacing

        # Tool results
        for tool_name, result in tool_results.items():
            if tool_name == 'history_context' or 'error' in result:
                continue

            if tool_name == 'get_courses':
                electives = result.get('electives', [])
                if electives:
                    summary_parts.append("AVAILABLE COURSES:")
                    for course in electives[:5]:
                        summary_parts.append(f"- {course}")

            elif tool_name == 'get_organizations':
                summary_parts.append("STUDENT ORGANIZATIONS:")
                for category, orgs in result.items():
                    if isinstance(orgs, list):
                        for org in orgs[:2]:
                            if isinstance(org, dict):
                                name = org.get('name', 'Unknown')
                                desc = org.get('description', '')
                                summary_parts.append(f"- {name}: {desc}")

            elif tool_name == 'get_companies':
                summary_parts.append("INTERNSHIPS AND CAREER OPPORTUNITIES:")
                for industry, companies in result.items():
                    if isinstance(companies, list):
                        for company in companies[:2]:
                            if isinstance(company, dict):
                                name = company.get('name', 'Unknown')
                                roles = ', '.join(company.get('opportunities', [])[:2])
                                summary_parts.append(f"- {name}: {roles}")

            elif tool_name == 'analyze_progress':
                missing_areas = result.get('missing_areas', [])
                if missing_areas:
                    summary_parts.append("UNEXPLORED AREAS:")
                    for area in missing_areas:
                        summary_parts.append(f"- {area}")

            elif tool_name == 'get_study_abroad':
                destinations = result.get('destinations', [])
                if destinations:
                    summary_parts.append("STUDY ABROAD OPPORTUNITIES:")
                    for d in destinations[:3]:
                        country = d.get('country', 'Unknown')
                        programs = ', '.join(d.get('programs', [])[:2])
                        summary_parts.append(f"- {country}: {programs}")

            elif tool_name == 'get_degree_requirements':
                reqs = result.get('requirements', [])
                if reqs:
                    summary_parts.append("DEGREE REQUIREMENTS:")
                    for r in reqs[:3]:
                        summary_parts.append(f"- {r}")

            elif tool_name == 'get_campus_facilities':
                facilities = result.get('facilities', [])
                if facilities:
                    summary_parts.append("CAMPUS FACILITIES:")
                    for f in facilities[:3]:
                        name = f.get('name', 'Unknown')
                        desc = f.get('description', '')
                        summary_parts.append(f"- {name}: {desc}")

            elif tool_name == 'get_university_system':
                structure = result.get('system_structure', [])
                if structure:
                    summary_parts.append("UNIVERSITY SYSTEM INFO:")
                    for s in structure[:3]:
                        summary_parts.append(f"- {s}")

            elif tool_name == 'get_academic_calendar':
                events = result.get('upcoming_events', [])
                if events:
                    summary_parts.append("ACADEMIC CALENDAR HIGHLIGHTS:")
                    for e in events[:3]:
                        name = e.get('name', 'Unknown')
                        date = e.get('date', '')
                        summary_parts.append(f"- {name} on {date}")

            elif tool_name == 'check_prerequisites':
                unmet = result.get('unmet_prerequisites', [])
                if unmet:
                    summary_parts.append("UNMET PREREQUISITES:")
                    for prereq in unmet[:3]:
                        summary_parts.append(f"- {prereq}")

        return "\n".join(summary_parts) if summary_parts else "No specific context available"

    def _parse_scenario_json(self, response: str) -> Dict[str, Any]:
        """Parse JSON scenario response with fallback"""
        try:
            # Try to parse the response as JSON
            parsed = json.loads(response.strip())
            return {
                "type": "life_choice",
                "title": parsed.get("title", "Life Choice"),
                "situation": parsed.get("situation", ""),
                "options": parsed.get("options", [])
            }
        except json.JSONDecodeError:
            # Try to extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                try:
                    json_str = response[start:end]
                    parsed = json.loads(json_str)
                    return {
                        "type": "life_choice",
                        "title": parsed.get("title", "Life Choice"),
                        "situation": parsed.get("situation", ""),
                        "options": parsed.get("options", [])
                    }
                except json.JSONDecodeError:
                    pass
            
            # Return empty dict to signal failure
            return {}

    def _parse_director_response(self, response: str) -> Dict[str, Any]:
        """Parse the director's analysis response"""
        result = {
            'analysis': '',
            'tools_to_use': [],
            'scenario_type': 'general'
        }
        
        logger.info("🎬 PARSING DIRECTOR RESPONSE:")
        logger.info(f"   Raw response length: {len(response)} characters")
        
        # Extract sections using more robust parsing
        lines = response.split('\n')
        current_section = None
        tools_section_lines = []
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('ANALYSIS:'):
                current_section = 'analysis'
                analysis_text = line.replace('ANALYSIS:', '').strip()
                if analysis_text:
                    result['analysis'] = analysis_text
            elif line.startswith('TOOLS_TO_USE:'):
                current_section = 'tools'
                # Check if tools are on the same line
                tools_text = line.replace('TOOLS_TO_USE:', '').strip()
                if tools_text:
                    tools_section_lines.append(tools_text)
            elif line.startswith('SCENARIO_TYPE:'):
                current_section = 'scenario'
                result['scenario_type'] = line.replace('SCENARIO_TYPE:', '').strip()
            elif current_section == 'analysis' and line:
                if result['analysis']:
                    result['analysis'] += '\n' + line
                else:
                    result['analysis'] = line
            elif current_section == 'tools' and line:
                tools_section_lines.append(line)
        
        # Process tools section
        for tool_line in tools_section_lines:
            # Handle multiple tools on one line separated by commas
            if ',' in tool_line:
                for tool in tool_line.split(','):
                    tool = tool.strip()
                    if tool:
                        result['tools_to_use'].append(tool)
            else:
                if tool_line.strip():
                    result['tools_to_use'].append(tool_line.strip())
        
        logger.info(f"   Parsed analysis: {len(result['analysis'])} chars")
        logger.info(f"   Parsed {len(result['tools_to_use'])} tools: {result['tools_to_use']}")
        logger.info(f"   Parsed scenario type: {result['scenario_type']}")
        
        return result
    
    def _parse_tool_call(self, tool_call: str) -> tuple:
        """Parse a tool call string like 'get_courses()' or 'get_organizations(interest_category=Technology)'"""
        logger.info(f"   Parsing tool call: {tool_call}")
        
        # Clean up the tool call - remove bullet points, dashes, and extra whitespace
        cleaned_call = tool_call.strip()
        if cleaned_call.startswith('-'):
            cleaned_call = cleaned_call[1:].strip()
        if cleaned_call.startswith('•'):
            cleaned_call = cleaned_call[1:].strip()
        
        if '(' in cleaned_call:
            tool_name = cleaned_call.split('(')[0].strip()
            params_str = cleaned_call.split('(')[1].rstrip(')')
            
            params = {}
            if params_str:
                # Simple parameter parsing - could be more robust
                for param in params_str.split(','):
                    if '=' in param:
                        key, value = param.split('=', 1)
                        params[key.strip()] = value.strip().strip('"\'')
            
            logger.info(f"   → Tool: {tool_name}, Params: {params}")
            return tool_name, params
        else:
            tool_name = cleaned_call.strip()
            logger.info(f"   → Tool: {tool_name}, No params")
            return tool_name, {}

# ==================== GLOBAL INSTANCES ====================

director = GameDirector(llm) if llm else None
game_sessions = {}

# ==================== API ROUTES ====================

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/api/start_game', methods=['POST'])
def start_game():
    """Start a new game session"""
    try:
        session_id = request.json.get('session_id', 'default')
        game_sessions[session_id] = PlayerProfile()
        logger.info(f"🎮 Started new game session: {session_id}")
        
        return jsonify({
            "success": True,
            "message": "Game started successfully",
            "profile": game_sessions[session_id].to_dict()
        })
    except Exception as e:
        logger.error(f"Error starting game: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/set_name', methods=['POST'])
def set_name():
    """Set player name"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        name = data.get('name', '').strip()
        
        if not name:
            return jsonify({"success": False, "error": "Name is required"}), 400
        
        if session_id not in game_sessions:
            game_sessions[session_id] = PlayerProfile()
        
        game_sessions[session_id].name = name
        logger.info(f"👤 Set name for session {session_id}: {name}")
        
        return jsonify({
            "success": True,
            "profile": game_sessions[session_id].to_dict()
        })
    except Exception as e:
        logger.error(f"Error setting name: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/generate_introduction', methods=['POST'])
def generate_introduction():
    """Generate personalized Chalmers introduction"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        
        if session_id not in game_sessions:
            return jsonify({"success": False, "error": "Session not found"}), 404
        
        profile = game_sessions[session_id]
        
        # Check if introduction was already generated
        if profile.has_seen_introduction and profile.introduction_text:
            return jsonify({
                "success": True,
                "introduction": profile.introduction_text,
                "cached": True
            })
        
        # Generate new introduction
        logger.info(f"🏫 Generating Chalmers introduction for {profile.name}")
        
        try:
            introduction_text = generate_chalmers_introduction(profile.name)
            
            # Store in profile
            profile.introduction_text = introduction_text
            profile.has_seen_introduction = True
            
            return jsonify({
                "success": True,
                "introduction": introduction_text,
                "cached": False
            })
            
        except Exception as intro_error:
            logger.error(f"Failed to generate introduction: {intro_error}")
            return jsonify({
                "success": False, 
                "error": f"Failed to generate introduction: {str(intro_error)}"
            }), 500
        
    except Exception as e:
        logger.error(f"Error in generate_introduction endpoint: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/set_program', methods=['POST'])
def set_program():
    """Set player program"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        program = data.get('program', '').strip()
        
        if not program:
            return jsonify({"success": False, "error": "Program is required"}), 400
        
        if session_id not in game_sessions:
            return jsonify({"success": False, "error": "Session not found"}), 404
        
        game_sessions[session_id].program = program
        game_sessions[session_id].current_situation = f"starting {program} at Chalmers"
        logger.info(f"🎓 Set program for session {session_id}: {program}")
        
        return jsonify({
            "success": True,
            "profile": game_sessions[session_id].to_dict()
        })
    except Exception as e:
        logger.error(f"Error setting program: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/generate_choice', methods=['POST'])
def generate_choice():
    """Generate next choice scenario using the AI Director"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        
        logger.info(f"🐛 DEBUG: Generate choice request for session: {session_id}")
        
        if session_id not in game_sessions:
            return jsonify({"success": False, "error": "Session not found"}), 404
        
        if not director:
            return jsonify({"success": False, "error": "AI director not available"}), 503
        
        profile = game_sessions[session_id]
        logger.info(f"🎮 API REQUEST: Generate choice for session {session_id}")
        
        # Use the AI Director to decide what happens next
        choice = director.decide_next_scenario(profile)
        
        logger.info(f"🎮 API RESPONSE: Successfully generated scenario for session {session_id}")
        
        return jsonify({
            "success": True,
            "choice": choice
        })
    except Exception as e:
        import traceback
        logger.error(f"🎮 API ERROR: Error generating choice: {e}")
        logger.error(f"🎮 API ERROR: Full traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/make_choice', methods=['POST'])
def make_choice():
    """Make a life choice"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        choice_index = data.get('choice_index')
        choice_data = data.get('choice_data')
        
        if session_id not in game_sessions:
            return jsonify({"success": False, "error": "Session not found"}), 404
        
        profile = game_sessions[session_id]
        
        if choice_data and 'options' in choice_data:
            options = choice_data['options']
            if 0 <= choice_index < len(options):
                selected_option = options[choice_index]
                
                # Check if this is a master's program selection
                if choice_data.get('type') == 'masters_selection':
                    # Extract program info from the option ID
                    option_id = selected_option.get('id', '')
                    if option_id.startswith('masters_'):
                        program_code = option_id.replace('masters_', '').upper()
                        # Find the full program info from metadata
                        metadata = choice_data.get('metadata', {})
                        all_programs = metadata.get('all_programs', [])
                        selected_program = next((p for p in all_programs if p.get('code') == program_code), None)
                        
                        if selected_program:
                            profile.selected_masters_program = selected_program['name']
                            logger.info(f"🎓 Selected master's program: {selected_program['name']}")
                            
                            # Force advance to year 4 after selecting master's
                            profile.year = 4
                            logger.info(f"📅 Advanced to Year 4 after master's selection")
                
                # Record the choice with full context
                choice_context = profile.add_choice(
                    choice_description=selected_option['description'],
                    choice_type=choice_data.get('title', 'Life decision'),
                    choice_data={
                        'situation': choice_data.get('situation', ''),
                        'options': choice_data.get('options', []),
                        'character_implication': selected_option.get('character_implication', '')
                    }
                )
                
                # IMPORTANT: Check year advancement AFTER adding choice (not during)
                # This ensures the year calculation is based on the new choice count
                year_advanced = False
                if choice_data.get('type') != 'masters_selection':  # Don't auto-advance if master's selection already advanced
                    year_advanced = profile.advance_year()
                    if year_advanced:
                        logger.info(f"📅 Year advanced to {profile.year} after choice")
                
                # Update personality
                if selected_option.get('character_implication'):
                    profile.personality_description = selected_option['character_implication']
                
                # Update profile summaries using AI if available
                if llm:
                    profile_service = ProfileUpdateService(llm)
                    profile_service.update_profile_summaries(profile, choice_context)
                
                logger.info(f"✅ Choice made for session {session_id}: {selected_option['description'][:50]}...")
                logger.info(f"   Updated personality: {profile.personality_description}")
                logger.info(f"   Current year: {profile.year}")
                logger.info(f"   Total choices: {len(profile.life_choices)}")
                
                return jsonify({
                    "success": True,
                    "profile": profile.to_dict(),
                    "selected_option": selected_option,
                    "year_advanced": choice_data.get('type') == 'masters_selection' or year_advanced  # Signal year advancement
                })
            else:
                return jsonify({"success": False, "error": "Invalid choice index"}), 400
        else:
            return jsonify({"success": False, "error": "Invalid choice data"}), 400
            
    except Exception as e:
        logger.error(f"Error making choice: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/get_profile', methods=['GET'])
def get_profile():
    """Get current player profile"""
    try:
        session_id = request.args.get('session_id', 'default')
        
        if session_id not in game_sessions:
            return jsonify({"success": False, "error": "Session not found"}), 404
        
        return jsonify({
            "success": True,
            "profile": game_sessions[session_id].to_dict()
        })
    except Exception as e:
        logger.error(f"Error getting profile: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# Debug/Admin endpoints
@app.route('/api/reload_knowledge', methods=['POST'])
def reload_knowledge():
    """Reload knowledge files"""
    try:
        knowledge.clear_cache()
        return jsonify({"success": True, "message": "Knowledge cache cleared"})
    except Exception as e:
        logger.error(f"Error reloading knowledge: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/debug/sessions', methods=['GET'])
def debug_sessions():
    """Get debug info about active sessions"""
    try:
        session_info = {}
        for session_id, profile in game_sessions.items():
            session_info[session_id] = {
                "name": profile.name,
                "program": profile.program,
                "choices_made": len(profile.life_choices),
                "current_year": profile.estimate_current_year(),
                "personality": profile.personality_description,
                "has_seen_introduction": profile.has_seen_introduction
            }
        
        return jsonify({
            "success": True,
            "active_sessions": len(game_sessions),
            "sessions": session_info
        })
    except Exception as e:
        logger.error(f"Error getting debug info: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/debug/director_tools', methods=['GET'])
def debug_director_tools():
    """Get info about available director tools"""
    try:
        if not director:
            return jsonify({"success": False, "error": "Director not available"}), 503
        
        tools_info = {}
        for tool_name, tool in director.tools.items():
            tools_info[tool_name] = {
                "name": tool.name,
                "description": tool.description
            }
        
        return jsonify({
            "success": True,
            "available_tools": len(director.tools),
            "tools": tools_info
        })
    except Exception as e:
        logger.error(f"Error getting director tools info: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/debug/status', methods=['GET'])
def debug_status():
    """Get current debug system status"""
    try:
        return jsonify({
            "success": True,
            "debug_logger_status": {
                "scenario_generations": len(debug_logger.scenario_generations),
                "llm_calls": len(debug_logger.llm_calls),
                "tool_executions": len(debug_logger.tool_executions),
                "current_scenario_active": debug_logger.current_scenario_gen is not None
            },
            "game_sessions": {
                "total_sessions": len(game_sessions),
                "session_ids": list(game_sessions.keys()),
                "default_session_exists": 'default' in game_sessions
            },
            "director_status": {
                "director_available": director is not None,
                "llm_available": llm is not None,
                "total_tools": len(director.tools) if director else 0
            }
        })
    except Exception as e:
        logger.error(f"Error getting debug status: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== DEBUG ENDPOINTS ====================

@app.route('/debug')
def debug_page():
    """Serve the debug interface HTML page"""
    try:
        # Read the debug HTML file
        with open('debug.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        # If debug.html doesn't exist, create it from your paste.txt content
        debug_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Debug Interface</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .error {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 4px;
            padding: 15px;
            margin: 10px 0;
            color: #721c24;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>AI Debug Interface</h1>
        <div class="error">
            Debug interface HTML not found. Please save your debug HTML as 'debug.html' in the project root.
        </div>
        <p>Available debug API endpoints:</p>
        <ul>
            <li><a href="/api/debug/player_profile">/api/debug/player_profile</a></li>
            <li><a href="/api/debug/last_scenario">/api/debug/last_scenario</a></li>
            <li><a href="/api/debug/ai_process">/api/debug/ai_process</a></li>
        </ul>
    </div>
</body>
</html>'''
        return debug_html

@app.route('/api/debug/last_scenario', methods=['GET'])
def debug_last_scenario():
    """Get detailed info about the last scenario generation"""
    try:
        if not debug_logger.scenario_generations:
            return jsonify({"success": False, "error": "No scenario generations found"}), 404
        
        last_generation = debug_logger.scenario_generations[-1]
        
        return jsonify({
            "success": True,
            "scenario_generation": asdict(last_generation)
        })
    except Exception as e:
        logger.error(f"Error getting last scenario debug data: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/debug/player_profile', methods=['GET'])
def debug_player_profile():
    """Get complete player profile with all fields"""
    try:
        session_id = request.args.get('session_id', 'default')
        
        # Add debugging info
        logger.info(f"🐛 DEBUG: Looking for session_id: {session_id}")
        logger.info(f"🐛 DEBUG: Available sessions: {list(game_sessions.keys())}")
        
        # If no sessions exist, create a default one for debugging
        if not game_sessions:
            logger.info("🐛 DEBUG: No sessions exist, creating default session")
            game_sessions['default'] = PlayerProfile()
            game_sessions['default'].name = "Debug User"
            game_sessions['default'].program = "Computer Science and Engineering"
        
        # If the requested session doesn't exist, use the first available one
        if session_id not in game_sessions:
            if game_sessions:
                session_id = next(iter(game_sessions.keys()))
                logger.info(f"🐛 DEBUG: Session not found, using: {session_id}")
            else:
                return jsonify({"success": False, "error": "No active sessions found"}), 404
        
        profile = game_sessions[session_id]
        
        # Get all fields dynamically
        profile_data = {}
        for field_name in profile.__dataclass_fields__:
            value = getattr(profile, field_name)
            
            # Handle sets and other non-serializable types
            if isinstance(value, set):
                profile_data[field_name] = list(value)
            else:
                profile_data[field_name] = value
        
        return jsonify({
            "success": True,
            "profile": profile_data,
            "field_types": {name: str(type(getattr(profile, name))) for name in profile.__dataclass_fields__},
            "session_id_used": session_id,
            "available_sessions": list(game_sessions.keys())
        })
    except Exception as e:
        logger.error(f"Error getting profile debug data: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/debug/test_logging', methods=['POST'])
def test_debug_logging():
    """Test if debug logging is working"""
    try:
        # Test logging a fake tool execution
        debug_logger.log_tool_execution(
            tool_name="test_tool",
            parameters={"test": "value"},
            raw_result={"result": "success"},
            success=True,
            processing_time=0.1
        )
        
        # Test logging a fake LLM call
        debug_logger.log_llm_call(
            call_type="test_call",
            prompt="Test prompt",
            model="test_model",
            max_tokens=100,
            raw_response="Test response",
            parsed_response={"content": "Test response"},
            success=True,
            processing_time=0.2
        )
        
        return jsonify({
            "success": True,
            "message": "Test logging completed",
            "total_tool_executions": len(debug_logger.tool_executions),
            "total_llm_calls": len(debug_logger.llm_calls)
        })
    except Exception as e:
        logger.error(f"Error testing debug logging: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/debug/ai_process', methods=['GET'])
def debug_ai_process():
    """Get detailed AI process debugging information"""
    try:
        session_id = request.args.get('session_id')
        limit = int(request.args.get('limit', 50))
        
        # Add debugging info
        logger.info(f"🐛 DEBUG: AI process request for session: {session_id}")
        logger.info(f"🐛 DEBUG: Total scenario generations: {len(debug_logger.scenario_generations)}")
        logger.info(f"🐛 DEBUG: Total LLM calls: {len(debug_logger.llm_calls)}")
        logger.info(f"🐛 DEBUG: Total tool executions: {len(debug_logger.tool_executions)}")
        
        debug_data = debug_logger.get_debug_data(session_id, limit)
        
        # Add extra debugging info to the response
        debug_data["debug_info"] = {
            "total_scenario_generations": len(debug_logger.scenario_generations),
            "total_llm_calls": len(debug_logger.llm_calls),
            "total_tool_executions": len(debug_logger.tool_executions),
            "current_scenario_gen": debug_logger.current_scenario_gen is not None,
            "session_id_requested": session_id
        }
        
        return jsonify({
            "success": True,
            "debug_data": debug_data
        })
    except Exception as e:
        logger.error(f"Error getting debug data: {e}")
        return jsonify({"success": False, "error": str(e)}), 500



# ==================== MAIN ====================

if __name__ == '__main__':
    logger.info("🚀 Starting Chalmers Life Journey with AI Director...")
    logger.info("📁 Validating knowledge files...")
    
    # Validate knowledge files exist - updated to match your actual files
    required_files = [
        'academic_calendar.json', 'campus_facilities.json', 'chalmers_courses.json',
        'current_events.json', 'gothenburg_companies.json', 'masters_programs.json',
        'programs.json', 'student_organizations.json', 'study_abroad_programs.json',
        'swedish_university_info.json'
    ]
    missing_files = []
    existing_files = []
    
    for filename in required_files:
        if not (Path("knowledge") / filename).exists():
            missing_files.append(filename)
        else:
            existing_files.append(filename)
    
    if existing_files:
        logger.info(f"✅ Found knowledge files: {', '.join(existing_files)}")
    
    if missing_files:
        logger.warning(f"⚠️  Missing knowledge files: {', '.join(missing_files)}")
        logger.info("The director will work with available files and provide fallbacks for missing data")
    else:
        logger.info("✅ All knowledge files found")
    
    logger.info("🎬 AI Director initialized with tools:")
    if director:
        for tool_name, tool in director.tools.items():
            logger.info(f"  - {tool_name}: {tool.description[:60]}{'...' if len(tool.description) > 60 else ''}")
        logger.info(f"🎬 Total tools available: {len(director.tools)}")
    
    logger.info("🌐 Server starting on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)