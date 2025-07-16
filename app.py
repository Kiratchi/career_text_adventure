from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Any, Optional, Set
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
    "chalmers_introduction": {"max_tokens": 400, "temperature": 0.7}
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

# ==================== SIMPLIFIED TOOL SYSTEM ====================

# Global tool registry
_tool_registry = {}

def register_tool(name: str):
    """Decorator to register tools automatically"""
    def decorator(cls):
        _tool_registry[name] = cls
        logger.info(f"🔧 Registered tool: {name} -> {cls.__name__}")
        return cls
    return decorator

class SimplifiedToolBase:
    """Base class for simplified tools"""
    
    def __init__(self, knowledge_loader):
        self.knowledge = knowledge_loader
    
    def execute(self, profile: 'PlayerProfile') -> str:
        """Execute the tool and return text content for LLM"""
        raise NotImplementedError
    
    def execute_with_debug(self, profile: 'PlayerProfile') -> str:
        """Execute tool with debugging (matches existing debug system)"""
        start_time = time.time()
        
        try:
            result = self.execute(profile)
            processing_time = time.time() - start_time
            
            debug_logger.log_tool_execution(
                tool_name=self.__class__.__name__,
                parameters={"student_program": profile.program, "student_year": profile.year},
                raw_result={"content_length": len(result), "content_preview": result[:200]},
                success=True,
                processing_time=processing_time
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            debug_logger.log_tool_execution(
                tool_name=self.__class__.__name__,
                parameters={"student_program": profile.program, "student_year": profile.year},
                raw_result={},
                success=False,
                error=str(e),
                processing_time=processing_time
            )
            
            raise

# ==================== SPECIFIC TOOLS ====================

@register_tool("sports")
class SportsTool(SimplifiedToolBase):
    """Tool for sports and physical activities information"""
    
    def execute(self, profile: 'PlayerProfile') -> str:
        try:
            sports_file = Path("knowledge") / "sports.txt"
            if not sports_file.exists():
                return "Sports information not available."
            
            with open(sports_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"🏃 Loaded sports.txt ({len(content)} characters)")
            return content
            
        except Exception as e:
            logger.error(f"Error loading sports.txt: {e}")
            return "Error loading sports information."

@register_tool("studies")
class StudiesTool(SimplifiedToolBase):
    """Tool for study environment, campus information, and student section details"""
    
    def execute(self, profile: 'PlayerProfile') -> str:
        try:
            # Load main studies information
            studies_file = Path("knowledge") / "studies.txt"
            if not studies_file.exists():
                return "Study information not available."
            
            with open(studies_file, 'r', encoding='utf-8') as f:
                studies_content = f.read()
            
            # Get student section information
            section_info = self._get_student_section_info(profile.program)
            
            # Combine the information
            output_lines = [
                "=== GENERAL STUDY INFORMATION ===",
                studies_content,
                "",
                "=== STUDENT SECTION INFORMATION ===",
                section_info
            ]
            
            result = "\n".join(output_lines)
            logger.info(f"📚 Loaded studies.txt ({len(studies_content)} characters) + student section info ({len(section_info)} characters)")
            return result
            
        except Exception as e:
            logger.error(f"Error loading study information: {e}")
            return "Error loading study information."
    
    def _get_student_section_info(self, program_code: str) -> str:
        """Get student section information for the given program"""
        # Program to student section mapping
        program_to_section = {
            "TKATK": "A",       # Architecture and Engineering
            "TKAUT": "Z",       # Automation and Mechatronics Engineering
            "TKBIO": "KfKb",    # Bioengineering
            "TKDAT": "D",       # Computer Science and Engineering
            "TKDES": "TD",      # Industrial Design Engineering
            "TKELT": "E",       # Electrical Engineering
            "TKGBS": "GS",      # Global Systems Engineering
            "TKIEK": "I",       # Industrial Engineering and Management
            "TKITE": "IT",      # Software Engineering
            "TKKEF": "KfKb",    # Chemical Eng. w/ Eng. Physics
            "TKKMT": "K",       # Chemical Engineering
            "TKMAS": "M",       # Mechanical Engineering
            "TKMED": "E",       # Biomedical Engineering
            "TKSAM": "V",       # Civil Engineering
            "TKTEM": "F",       # Engineering Mathematics
            "TKTFY": "F",       # Engineering Physics
            "TKARK": "A",       # Architecture
        }
        
        # Get the student section code
        section_code = program_to_section.get(program_code)
        if not section_code:
            return f"No student section information found for program {program_code}"
        
        # Try to load the student section file
        try:
            section_file = Path("knowledge") / "student_sections" / f"{section_code}.txt"
            if not section_file.exists():
                logger.warning(f"Student section file not found: {section_file}")
                return f"Student section information not available for section {section_code}"
            
            with open(section_file, 'r', encoding='utf-8') as f:
                section_content = f.read()
            
            logger.info(f"📋 Loaded student section {section_code}.txt ({len(section_content)} characters)")
            return f"Student Section {section_code} (for {program_code}):\n\n{section_content}"
            
        except Exception as e:
            logger.error(f"Error loading student section {section_code}: {e}")
            return f"Error loading student section information for {section_code}"

@register_tool("exchange")
class ExchangeTool(SimplifiedToolBase):
    """Tool for international exchange opportunities during master's program"""
    
    def execute(self, profile: 'PlayerProfile') -> str:
        try:
            exchange_file = Path("knowledge") / "exchange.txt"
            if not exchange_file.exists():
                return "Exchange information not available."
            
            with open(exchange_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"✈️ Loaded exchange.txt ({len(content)} characters)")
            return content
            
        except Exception as e:
            logger.error(f"Error loading exchange.txt: {e}")
            return "Error loading exchange information."

@register_tool("courses")
class CoursesTool(SimplifiedToolBase):
    """Tool for courses available to the student based on their program and year"""
    
    def execute(self, profile: 'PlayerProfile') -> str:
        try:
            # Load courses for the student's program
            courses_in_program = self.knowledge.load_json('courses_in_program.json')
            course_summaries = self.knowledge.load_json('course_summary_simplified.json')
            
            if not courses_in_program or not course_summaries:
                return "Course information not available."
            
            # Get program code - if it's already a code, use it. If it's a name, convert it.
            program_code = self._get_program_code(profile.program)
            if not program_code:
                return f"Program code not found for {profile.program}"
            
            # Get courses for this program
            program_courses = courses_in_program.get('programs', {}).get(program_code, {})
            if not program_courses:
                return f"No courses found for program {program_code}"
            
            # Create course summary lookup
            course_lookup = {course['courseCode']: course for course in course_summaries}
            
            # Build output text
            output_lines = [
                f"COURSES AVAILABLE FOR {profile.program} (Program Code: {program_code})",
                f"Student Year: {profile.year}",
                "",
                "COURSE DETAILS:",
                ""
            ]
            
            # Sort courses by code for consistency
            for course_code in sorted(program_courses.keys()):
                study_periods = program_courses[course_code]
                course_info = course_lookup.get(course_code)
                
                if course_info:
                    output_lines.extend([
                        f"Course: {course_code} - {course_info['name']}",
                        f"Study Periods: {', '.join(study_periods)}",
                        f"Description: {course_info['AI_summary']}",
                        ""
                    ])
                else:
                    output_lines.extend([
                        f"Course: {course_code}",
                        f"Study Periods: {', '.join(study_periods)}",
                        "Description: Course details not available",
                        ""
                    ])
            
            result = "\n".join(output_lines)
            logger.info(f"📝 Generated course information for {program_code}: {len(program_courses)} courses")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating course information: {e}")
            return f"Error loading course information: {str(e)}"
    
    def _get_program_code(self, program_identifier: str) -> str:
        """Get program code - if it's already a code, return it. If it's a name, convert it."""
        # If it's already a program code (starts with TK or MP), return as-is
        if program_identifier.startswith(('TK', 'MP', 'TA', 'TI', 'TS', 'KP')):
            return program_identifier
        
        # Program name to code mapping for backward compatibility
        program_mapping = {
            # Base programs (Master of Science in Engineering)
            "Architecture and Engineering": "TKATK",
            "Automation and Mechatronics Engineering": "TKAUT", 
            "Bioengineering": "TKBIO",
            "Computer Science and Engineering": "TKDAT",
            "Industrial Design Engineering": "TKDES",
            "Electrical Engineering": "TKELT",
            "Global Systems Engineering": "TKGBS",
            "Industrial Engineering and Management": "TKIEK",
            "Software Engineering": "TKITE",
            "Chemical Engineering With Engineering Physics": "TKKEF",
            "Chemical Engineering": "TKKMT",
            "Mechanical Engineering": "TKMAS",
            "Biomedical Engineering": "TKMED",
            "Civil Engineering": "TKSAM",
            "Engineering Mathematics": "TKTEM",
            "Engineering Physics": "TKTFY",
            "Architecture": "TKARK",
            
            # Master programs (Master of Science) - common names
            "Materials Engineering": "MPAEM",
            "Computer Science": "MPALG",
            "Applied Mechanics": "MPAME",
            "Data Science and AI": "MPDSC",
            "Software Engineering and Technology": "MPSOF",
            # Add more as needed
        }
        
        return program_mapping.get(program_identifier, program_identifier)

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

class ProgramMappingService:
    """Centralized service for mapping program codes to names and descriptions"""
    
    def __init__(self, knowledge_loader):
        self.knowledge = knowledge_loader
        self._program_cache = {}
        self._master_cache = {}
    
    def get_bachelor_program_info(self, program_code: str) -> Dict[str, str]:
        """Get bachelor program information by code"""
        if program_code in self._program_cache:
            return self._program_cache[program_code]
        
        # Load from programs.json
        programs_data = self.knowledge.load_json('programs.json')
        if programs_data:
            # Use the correct structure: programs_data['programs']
            programs_list = programs_data.get('programs', [])
            for program in programs_list:
                code = program.get('code', '')
                if code == program_code:
                    result = {
                        'code': code,
                        'name': program.get('name', f'Program {code}'),
                        'explanation': program.get('explanation', 'No description available')
                    }
                    self._program_cache[program_code] = result
                    return result
        
        # If not found, return minimal info
        logger.warning(f"Program {program_code} not found in programs.json")
        result = {
            'code': program_code,
            'name': f'Program {program_code}',
            'explanation': 'Program information not available'
        }
        self._program_cache[program_code] = result
        return result
    
    def get_master_program_info(self, master_code: str) -> Dict[str, str]:
        """Get master's program information by code"""
        if master_code in self._master_cache:
            return self._master_cache[master_code]
        
        # Load from masters_programs.json
        masters_data = self.knowledge.load_json('masters_programs.json')
        if masters_data:
            # Use the correct structure: masters_data['programs']
            programs_list = masters_data.get('programs', [])
            for program in programs_list:
                code = program.get('code', '')
                if code == master_code:
                    result = {
                        'code': code,
                        'name': program.get('name', f'Master\'s Program {code}'),
                        'description': program.get('explanation', 'Advanced studies in this field')
                    }
                    self._master_cache[master_code] = result
                    return result
        
        # If not found, return minimal info
        logger.warning(f"Master's program {master_code} not found in masters_programs.json")
        result = {
            'code': master_code,
            'name': f'Master\'s Program {master_code}',
            'description': 'Program information not available'
        }
        self._master_cache[master_code] = result
        return result
    
    def get_available_masters_for_bachelor(self, bachelor_code: str) -> List[Dict[str, str]]:
        """Get all available master's programs for a given bachelor program"""
        # Load the bidirectional mapping
        mapping_data = self.knowledge.load_json('program_master_bidirectional_mapping.json')
        if not mapping_data:
            logger.warning("Could not load master's program mapping")
            return []
        
        programs_to_masters = mapping_data.get('programs_to_masters', {})
        available_master_codes = programs_to_masters.get(bachelor_code, [])
        
        # Get full info for each master's program
        master_programs = []
        for master_code in available_master_codes:
            master_info = self.get_master_program_info(master_code)
            master_programs.append(master_info)
        
        return master_programs
    
    def get_program_code_from_name(self, program_name: str) -> str:
        """Get program code from program name (for backward compatibility)"""
        # Load all programs and find matching name
        programs_data = self.knowledge.load_json('programs.json')
        if programs_data:
            # Use the correct structure: programs_data['programs']
            programs_list = programs_data.get('programs', [])
            for program in programs_list:
                if program.get('name') == program_name:
                    return program.get('code', program_name)
        
        # If not found, log warning and return the name as-is
        logger.warning(f"Program name '{program_name}' not found in programs.json")
        return program_name
    
    def clear_cache(self):
        """Clear the program mapping cache"""
        self._program_cache.clear()
        self._master_cache.clear()
        logger.info("🧹 Program mapping cache cleared")

# Global knowledge loader
knowledge = KnowledgeLoader()
program_mapping = ProgramMappingService(knowledge)

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

# ==================== PROGRAM EXPLANATION HELPER ====================

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

# ==================== GAME DIRECTOR ====================

class GameDirector:
    """Enhanced Game Director with simplified unified scenario generation"""
    
    def __init__(self, llm, knowledge_loader):
        self.llm = llm
        self.knowledge = knowledge_loader
        self.tools = self._initialize_tools()
        self._generating_scenario = False
        self._current_ai_analysis = {}
    
    def _initialize_tools(self) -> Dict[str, Any]:
        """Initialize all registered tools"""
        tools = {}
        for tool_name, tool_class in _tool_registry.items():
            tools[tool_name] = tool_class(self.knowledge)
            logger.info(f"🔧 Initialized tool: {tool_name}")
        
        logger.info(f"🎬 Director initialized with {len(tools)} tools")
        return tools
    
    def _execute_tool(self, tool_name: str, profile: Any) -> str:
        """Execute a specific tool"""
        if tool_name not in self.tools:
            logger.warning(f"🔧 Tool '{tool_name}' not found")
            return f"Tool '{tool_name}' not available"
        
        try:
            logger.info(f"🔧 EXECUTING TOOL: {tool_name}")
            logger.info(f"   Student: {profile.name} ({profile.program}, Year {profile.year})")
            
            result = self.tools[tool_name].execute_with_debug(profile)
            
            logger.info(f"   → Success: {len(result)} characters returned")
            return result
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return f"Error executing {tool_name}: {str(e)}"

    def decide_next_scenario(self, profile: Any) -> Dict[str, Any]:
        """Enhanced scenario decision with simplified unified approach"""
        
        # Add recursion prevention
        if hasattr(self, '_generating_scenario') and self._generating_scenario:
            logger.warning("🎬 RECURSION DETECTED: Already generating scenario, returning fallback")
            return self._create_ai_crash_fallback(profile)
        
        self._generating_scenario = True
        
        try:
            # Get the session_id from the request context or use default
            session_id = getattr(request, 'json', {}).get('session_id', 'default') if 'request' in globals() else 'default'
            
            # Start debug tracking
            debug_logger.start_scenario_generation(session_id, profile)
            
            logger.info(f"🎬 DIRECTOR STARTING: Decision for {profile.name} ({profile.program}, Year {profile.year})")
            
            # Check if student should advance year first
            year_advanced = profile.advance_year()
            if year_advanced:
                logger.info(f"📅 Student advanced to Year {profile.year}")
            
            # STEP 1: DETERMINE SCENARIO TYPE
            choices_count = len(profile.life_choices)
            scenario_type = self._determine_scenario_type(profile, choices_count)
            
            logger.info(f"🎯 SCENARIO TYPE: {scenario_type}")
            
            # STEP 2: GENERATE SCENARIO
            scenario = self._generate_scenario(profile, scenario_type)
            
            debug_logger.complete_scenario_generation(scenario, True, total_time=0.0)
            return scenario
            
        except Exception as e:
            start_time = time.time()  # Define start_time for the error case
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

    def _determine_scenario_type(self, profile: Any, choices_count: int) -> str:
        """STEP 1: Determine what type of scenario to create"""
        
        # MOTTAGNING - First choice of Year 1 (0 choices)
        if choices_count == 0 and profile.year == 1:
            logger.info("🎉 SCENARIO TYPE: Mottagning (First choice of Year 1)")
            return "mottagning"
        
        # Master's program selection (after 5 choices = end of year 3)
        if choices_count == 5 and not profile.selected_masters_program:
            logger.info("🎓 SCENARIO TYPE: Master's Program Selection (5 choices)")
            return "masters_selection"
        
        # Exchange opportunity (50% chance at 6 or 8 choices during master's)
        if choices_count in [6, 8] and profile.year >= 4 and not self._has_done_exchange(profile):
            if self._roll_exchange_chance(profile, choices_count):
                logger.info(f"✈️ SCENARIO TYPE: Exchange (50% chance at {choices_count} choices)")
                return "exchange"
        
        # Thesis project selection (at 9 choices = mid year 5)
        if choices_count == 9 and not self._has_done_thesis(profile):
            logger.info("📝 SCENARIO TYPE: Thesis Project Selection (9 choices)")
            return "thesis"
        
        # Career preparation (at 10 choices = late year 5)
        if choices_count == 10:
            logger.info("💼 SCENARIO TYPE: Career Preparation (10 choices)")
            return "career"
        
        # Graduation scenario (at 11+ choices)
        if choices_count >= 11:
            logger.info("🎓 SCENARIO TYPE: Graduation (11+ choices)")
            return "graduation"
        
        # DEFAULT: Generate AI-driven scenario for regular choices
        logger.info(f"🤖 SCENARIO TYPE: Standard AI (default for {choices_count} choices)")
        return "standard"

    def _generate_scenario(self, profile: Any, scenario_type: str) -> Dict[str, Any]:
        """STEP 2: Generate the scenario based on type - UNIFIED WORKFLOW"""
        
        if not self.llm:
            logger.warning(f"🎬 LLM not available for {scenario_type}, using fallback")
            return self._create_ai_crash_fallback(profile)
        
        try:
            start_time = time.time()
            
            # STEP 2.5: Get tools and prompt for this scenario type
            tools_to_use, prompt_name = self._get_scenario_config(scenario_type)
            
            logger.info(f"🎬 SCENARIO CONFIG: prompt='{prompt_name}', tools={tools_to_use}")
            
            # STEP 3: Execute tools
            tool_results = self._execute_tools(tools_to_use, profile)
            
            # STEP 2 (continued): Get high-level analysis
            analysis = self._get_scenario_analysis(profile, prompt_name)
            
            # STEP 4: Generate scenario (ALWAYS using the same code)
            scenario = self._create_scenario_with_context_retry(
                profile, 
                analysis.get('scenario_type', 'general'),
                analysis.get('analysis', ''),
                tool_results
            )
            
            total_time = time.time() - start_time
            
            debug_logger.complete_scenario_generation(
                final_scenario=scenario,
                success=True,
                total_time=total_time
            )
            
            logger.info(f"🎬 SCENARIO COMPLETE: '{scenario.get('title', 'Unknown')}' with {len(scenario.get('options', []))} options")
            return scenario
            
        except Exception as e:
            logger.error(f"🎬 Error generating {scenario_type} scenario: {e}")
            return self._create_ai_crash_fallback(profile)

    def _get_scenario_config(self, scenario_type: str) -> Tuple[List[str], str]:
        """STEP 2.5: Get tools and prompt name for each scenario type"""
        
        scenario_configs = {
            "mottagning": (["studies"], "mottagning_analysis"),
            "masters_selection": ([], "masters_introduction_analysis"),
            "exchange": (["exchange"], "exchange_analysis"),
            "thesis": (["studies", "courses"], "thesis_analysis"),
            "career": ([], "career_analysis"),
            "graduation": ([], "graduation_analysis"),
            "standard": (None, "director_analysis")  # Special case - AI decides tools
        }
        
        return scenario_configs.get(scenario_type, ([], "director_analysis"))

    def _execute_tools(self, tools_to_use: List[str], profile: Any) -> Dict[str, str]:
        """STEP 3: Execute the tools"""
        
        if tools_to_use is None:
            # Special case for standard scenarios - let AI decide tools
            return self._execute_ai_chosen_tools(profile)
        
        # Execute hardcoded tools
        tool_results = {}
        for tool_name in tools_to_use:
            if tool_name in self.tools:
                tool_results[tool_name] = self._execute_tool(tool_name, profile)
            else:
                logger.warning(f"   → Tool not found: {tool_name}")
        
        logger.info(f"🔧 TOOLS EXECUTED: {len(tool_results)} tools completed")
        return tool_results

    def _execute_ai_chosen_tools(self, profile: Any) -> Dict[str, str]:
        """Execute tools chosen by AI analysis (for standard scenarios)"""
        
        logger.info("🤖 AI TOOL SELECTION: Getting AI analysis for tool selection...")
        
        # Get AI analysis to determine tools
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
            logger.error("🤖 AI TOOL SELECTION: Director analysis prompt failed to load!")
            return {}
        
        # Make AI call to get tool selection
        messages = [{"role": "user", "content": prompt_content}]
        response = self._make_llm_call_with_retry(
            messages, 
            call_type="director_analysis",
            prompt_name="director_analysis"
        )
        
        if response is None:
            logger.error("🤖 AI TOOL SELECTION: AI analysis failed")
            return {}
        
        # Parse AI response to get tools
        analysis = self._parse_director_response(response)
        
        logger.info("🤖 AI ANALYSIS:")
        logger.info(f"   Reasoning: {analysis.get('analysis', 'No analysis provided')[:100]}...")
        logger.info(f"   Tools Planned: {len(analysis.get('tools_to_use', []))} tools")
        
        # Execute the tools AI requested
        tool_results = {}
        for i, tool_call in enumerate(analysis.get('tools_to_use', []), 1):
            logger.info(f"🔧 AI TOOL {i}/{len(analysis.get('tools_to_use', []))}: {tool_call}")
            tool_name = self._parse_tool_call(tool_call)
            if tool_name and tool_name in self.tools:
                tool_results[tool_name] = self._execute_tool(tool_name, profile)
            else:
                logger.warning(f"   → Tool not found or could not parse: {tool_call}")
        
        # Store analysis for later use
        self._current_ai_analysis = analysis
        
        return tool_results

    def _get_scenario_analysis(self, profile: Any, prompt_name: str) -> Dict[str, Any]:
        """Get high-level analysis for scenario creation"""
        
        if prompt_name == "director_analysis":
            # For standard scenarios, we already have the analysis
            return getattr(self, '_current_ai_analysis', {
                'analysis': 'Standard scenario analysis',
                'scenario_type': 'general'
            })
        else:
            # For specific scenarios, get analysis from the specific prompt
            logger.info(f"🎬 Getting analysis from {prompt_name}")
            
            try:
                analysis_prompt_content = load_prompt(
                    prompt_name,
                    student_name=profile.name,
                    program=profile.program,
                    year=profile.year,
                    program_explanation=GetProgramsExplanation.get_explanation(profile.program),
                    comprehensive_history=profile.comprehensive_history or "This is their first significant decision at university.",
                    personality_summary=profile.personality_summary or profile.personality_description,
                    masters_program=profile.selected_masters_program or "their chosen specialization",
                    available_masters=self._get_available_masters_list(profile.program)
                )
                
                if not analysis_prompt_content:
                    logger.error(f"🎬 ERROR: Analysis prompt content is empty for {prompt_name}!")
                    return {'analysis': 'Analysis failed', 'scenario_type': 'general'}
                
                # Make LLM call for analysis
                messages = [{"role": "user", "content": analysis_prompt_content}]
                analysis_response = self._make_llm_call_with_retry(
                    messages, 
                    call_type=f"{prompt_name}_analysis",
                    prompt_name=prompt_name
                )
                
                if analysis_response is None:
                    logger.error(f"🎬 {prompt_name.upper()}: Analysis failed after all retries")
                    return {'analysis': 'Analysis failed', 'scenario_type': 'general'}
                
                # Parse the analysis response
                analysis = self._parse_director_response(analysis_response)
                
                logger.info(f"🎬 ANALYSIS COMPLETE:")
                logger.info(f"   Analysis: {analysis.get('analysis', 'No analysis provided')[:100]}...")
                logger.info(f"   Scenario Type: {analysis.get('scenario_type', 'unknown')}")
                
                return analysis
                
            except Exception as e:
                logger.error(f"🎬 Error getting analysis from {prompt_name}: {e}")
                return {'analysis': 'Analysis failed', 'scenario_type': 'general'}

    def _get_available_masters_list(self, program_code: str) -> str:
        """Get a formatted list of available master's programs for prompts"""
        try:
            available_masters = program_mapping.get_available_masters_for_bachelor(program_code)
            if available_masters:
                return ', '.join([f"{m['name']} ({m['code']})" for m in available_masters[:8]])
            return "No master's programs available"
        except:
            return "Master's programs information not available"

    def _get_tool_descriptions(self) -> str:
        """Get descriptions of all available tools for LLM prompts"""
        descriptions = []
        for tool_name, tool in self.tools.items():
            description = tool.__class__.__doc__ or f"Tool for {tool_name}"
            descriptions.append(f"- {tool_name}: {description.strip()}")
        
        return "\n".join(descriptions)

    def _create_ai_crash_fallback(self, profile: Any) -> Dict[str, Any]:
        """Create a humorous fallback when the AI has completely crashed"""
        
        crash_situations = [
            f"The AI Director has blue-screened while trying to figure out what {profile.name} should do next. In the confusion, three random choices have appeared:",
            f"ERROR 404: Plot not found. The AI Director is currently stuck in an infinite loop thinking about {profile.name}'s future. Please choose from these emergency backup options:",
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
            }
        ]
        
        situation = random.choice(crash_situations)
        
        return {
            "type": "ai_crash",
            "title": "🤖 AI DIRECTOR MALFUNCTION 🤖",
            "situation": situation,
            "options": crash_options
        }

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
                
                if attempt == max_retries - 1:
                    logger.error(f"🤖 LLM CALL FAILED after {max_retries} attempts")
                    break
                
                wait_time = 0.5 * (2 ** attempt)
                logger.info(f"🤖 Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        
        return None

    def _parse_director_response(self, response: str) -> Dict[str, Any]:
        """Parse the director's analysis response"""
        result = {
            'analysis': '',
            'tools_to_use': [],
            'scenario_type': 'general'
        }
        
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
            if ',' in tool_line:
                for tool in tool_line.split(','):
                    tool = tool.strip()
                    if tool:
                        result['tools_to_use'].append(tool)
            else:
                if tool_line.strip():
                    result['tools_to_use'].append(tool_line.strip())
        
        return result
    
    def _parse_tool_call(self, tool_call: str) -> str:
        """Parse a tool call string and return just the tool name"""
        # Clean up the tool call
        cleaned_call = tool_call.strip()
        if cleaned_call.startswith('-'):
            cleaned_call = cleaned_call[1:].strip()
        if cleaned_call.startswith('•'):
            cleaned_call = cleaned_call[1:].strip()
        
        # Remove parentheses and parameters - we don't use them anymore
        if '(' in cleaned_call:
            tool_name = cleaned_call.split('(')[0].strip()
        else:
            tool_name = cleaned_call.strip()
        
        return tool_name
    
    def _create_scenario_with_context_retry(self, profile: Any, scenario_type: str, analysis: str, tool_results: Dict[str, str]) -> Dict[str, Any]:
        """Create the final scenario using AI with context from tools"""
        
        logger.info("🎬 PHASE 5: Creating final scenario with AI...")
        
        context_summary = self._summarize_tool_results(tool_results)
        
        try:
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
                activities_done='',  # You may want to implement this
                organizations_joined='',  # You may want to implement this
                courses_taken='',  # You may want to implement this
                companies_applied='',  # You may want to implement this
                repetition_warnings='No repetition concerns',
                variety_suggestions='Continue exploring new areas',
                context_summary=context_summary
            )
            
            if not prompt_content:
                logger.error("🎬 ERROR: Prompt content is empty!")
                return self._create_ai_crash_fallback(profile)
            
            # Make LLM call with retry
            messages = [{"role": "user", "content": prompt_content}]
            response = self._make_llm_call_with_retry(
                messages, 
                call_type="scenario_creation",
                prompt_name="scenario_creation"
            )
            
            if response is None:
                logger.error("🎬 SCENARIO CREATION: AI failed after all retries")
                return self._create_ai_crash_fallback(profile)
            
            # Parse JSON response
            scenario = self._parse_scenario_json(response)
            
            if not scenario.get('options'):
                logger.error("🎬 ERROR: Scenario parsing failed or no options found")
                return self._create_ai_crash_fallback(profile)
            
            return scenario
            
        except Exception as e:
            logger.error(f"🎬 Error creating scenario: {e}")
            return self._create_ai_crash_fallback(profile)
    
    def _summarize_tool_results(self, tool_results: Dict[str, str]) -> str:
        """Summarize tool results for scenario prompt context"""
        if not tool_results:
            return "No tool data available"
        
        summary_parts = []
        for tool_name, result in tool_results.items():
            summary_parts.append(f"=== {tool_name.upper()} ===")
            # Include full results - let the LLM handle the full context
            summary_parts.append(result)
            summary_parts.append("")
        
        return "\n".join(summary_parts)
    
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

    # HELPER METHODS
    def _has_done_exchange(self, profile: Any) -> bool:
        """Check if student has already done exchange"""
        exchange_keywords = ['exchange', 'abroad', 'international', 'erasmus']
        return any(any(keyword in choice.lower() for keyword in exchange_keywords) 
                   for choice in profile.life_choices)

    def _has_done_thesis(self, profile: Any) -> bool:
        """Check if student has already done thesis project"""
        thesis_keywords = ['thesis', 'dissertation', 'research project', 'final project']
        return any(any(keyword in choice.lower() for keyword in thesis_keywords) 
                   for choice in profile.life_choices)

    def _roll_exchange_chance(self, profile: Any, choices_count: int) -> bool:
        """50% chance for exchange, using student name + choices as seed for consistency"""
        import hashlib
        seed = hashlib.md5(f"{profile.name}_exchange_{choices_count}".encode()).hexdigest()
        return int(seed[:8], 16) % 2 == 0

# ==================== GLOBAL INSTANCES ====================

director = GameDirector(llm, knowledge) if llm else None
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

# ==================== TEST TOOLS ENDPOINT ====================

@app.route('/api/test_tools', methods=['GET'])
def test_tools():
    """Test the new tool system"""
    try:
        if not director:
            return jsonify({"success": False, "error": "Director not available"})
        
        # Create a test profile
        test_profile = PlayerProfile()
        test_profile.name = "Test Student"
        test_profile.program = "TKDAT"  # Computer Science and Engineering
        test_profile.year = 2
        
        # Test each tool
        results = {}
        for tool_name in director.tools.keys():
            try:
                result = director._execute_tool(tool_name, test_profile)
                results[tool_name] = {
                    "success": True,
                    "content_length": len(result),
                    "preview": result[:200] + "..." if len(result) > 200 else result
                }
            except Exception as e:
                results[tool_name] = {
                    "success": False,
                    "error": str(e)
                }
        
        return jsonify({
            "success": True,
            "available_tools": len(director.tools),
            "tool_results": results
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

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
                "name": tool.__class__.__name__,
                "description": tool.__class__.__doc__ or f"Tool for {tool_name}"
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
            <li><a href="/api/test_tools">/api/test_tools</a></li>
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
            game_sessions['default'].program = "TKDAT"  # Use program code
        
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

@app.route('/api/debug/quick_setups', methods=['GET'])
def debug_quick_setups():
    """Get predefined quick setups with exact choice targeting"""
    try:
        quick_setups = {
            "pre_masters": {
                "description": "Pre-Master's Selection (4 choices, Year 3)",
                "year": 3,
                "choices": [
                    "Academic Focus: Completed core engineering courses with high grades",
                    "Social Growth: Joined student section and made lasting friendships", 
                    "Technical Skills: Participated in programming competitions",
                    "Career Prep: Completed summer internship at local tech company"
                ]
            },
            "trigger_masters": {
                "description": "Trigger Master's Selection (5 choices, Year 3)",
                "year": 3,
                "choices": [
                    "Academic Focus: Excelled in fundamental engineering courses",
                    "Social Growth: Became active in student organizations",
                    "Technical Skills: Developed strong programming skills",
                    "Career Prep: Explored different career paths through internships",
                    "Academic Achievement: Maintained high GPA and research interests"
                ]
            },
            "trigger_exchange_6": {
                "description": "Trigger Exchange at 6 choices (Year 4)",
                "year": 4,
                "choices": [
                    "Academic Focus: Completed bachelor's with excellent results",
                    "Social Growth: Built strong network at Chalmers",
                    "Technical Skills: Specialized in chosen field",
                    "Career Prep: Gained industry experience",
                    "Masters Selection: Chose advanced master's program",
                    "Research Focus: Started master's level coursework"
                ]
            },
            "trigger_exchange_8": {
                "description": "Trigger Exchange at 8 choices (Year 4)", 
                "year": 4,
                "choices": [
                    "Academic Focus: Completed bachelor's degree successfully",
                    "Social Growth: Developed leadership skills in student section",
                    "Technical Skills: Mastered core technical competencies", 
                    "Career Prep: Built professional network through internships",
                    "Masters Selection: Selected specialized master's program",
                    "Research Focus: Engaged with advanced research topics",
                    "Academic Excellence: Achieved outstanding results in master's courses",
                    "Professional Development: Gained deeper industry insights"
                ]
            },
            "trigger_thesis": {
                "description": "Trigger Thesis Selection (9 choices, Year 5)",
                "year": 5,
                "choices": [
                    "Academic Focus: Excellence throughout undergraduate studies",
                    "Social Growth: Leadership roles in student community",
                    "Technical Skills: Advanced technical expertise",
                    "Career Prep: Strong industry connections",
                    "Masters Selection: Specialized master's program",
                    "Research Focus: Deep research engagement", 
                    "Academic Achievement: Outstanding master's coursework",
                    "Professional Growth: Industry collaboration experience",
                    "Research Excellence: Ready for thesis-level work"
                ]
            },
            "trigger_career": {
                "description": "Trigger Career Preparation (10 choices, Year 5)",
                "year": 5,
                "choices": [
                    "Academic Focus: Maintained excellence throughout studies",
                    "Social Growth: Strong leadership and community involvement",
                    "Technical Skills: Expert-level technical competencies",
                    "Career Prep: Extensive industry network and experience",
                    "Masters Selection: Advanced specialization completed",
                    "Research Focus: Significant research contributions",
                    "Academic Achievement: Top-tier academic performance",
                    "Professional Growth: Industry mentorship and guidance",
                    "Research Excellence: Thesis project well underway",
                    "Career Readiness: Prepared for professional transition"
                ]
            },
            "trigger_graduation": {
                "description": "Trigger Graduation (11+ choices, Year 5)",
                "year": 5,
                "choices": [
                    "Academic Focus: Consistently high academic performance",
                    "Social Growth: Extensive leadership and community impact",
                    "Technical Skills: Industry-ready technical expertise",
                    "Career Prep: Multiple career opportunities secured",
                    "Masters Selection: Specialized expertise fully developed",
                    "Research Focus: Published research and thesis completion",
                    "Academic Achievement: Recognition for academic excellence",
                    "Professional Growth: Strong industry relationships",
                    "Research Excellence: Thesis defense preparation",
                    "Career Readiness: Job offers and career planning",
                    "Final Preparations: Ready for graduation and next chapter"
                ]
            }
        }
        
        return jsonify({
            "success": True,
            "quick_setups": quick_setups
        })
    except Exception as e:
        logger.error(f"Error getting quick setups: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    
@app.route('/api/debug/apply_quick_setup', methods=['POST'])
def debug_apply_quick_setup():
    """Apply a predefined quick setup to a session"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        setup_name = data.get('setup_name')
        
        if not setup_name:
            return jsonify({"success": False, "error": "setup_name is required"}), 400
        
        # Get the quick setup data
        quick_setups_response = debug_quick_setups()
        quick_setups_data = quick_setups_response.get_json()
        
        if not quick_setups_data.get('success'):
            return jsonify({"success": False, "error": "Failed to load quick setups"}), 500
        
        setup = quick_setups_data['quick_setups'].get(setup_name)
        if not setup:
            return jsonify({"success": False, "error": f"Setup '{setup_name}' not found"}), 404
        
        # Apply the setup
        return debug_jump_to_state_internal(session_id, setup['year'], setup['choices'])
        
    except Exception as e:
        logger.error(f"Error applying quick setup: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/debug/jump_to_state', methods=['POST'])
def debug_jump_to_state():
    """Jump to a specific game state"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        year = data.get('year', 1)
        choices = data.get('choices', [])
        
        return debug_jump_to_state_internal(session_id, year, choices)
        
    except Exception as e:
        logger.error(f"Error jumping to state: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    
def calculate_choices_for_year(target_year):
    """Calculate how many choices should be made by a certain year"""
    # Year progression: 2 choices per year, starting from 0
    # Year 1: 0-1 choices
    # Year 2: 2-3 choices  
    # Year 3: 4-5 choices (master's selection at 5)
    # Year 4: 6-7 choices (exchange at 6 or 8)
    # Year 5: 8+ choices
    
    if target_year <= 1:
        return 1  # Early year 1
    elif target_year == 2:
        return 3  # Mid year 2
    elif target_year == 3:
        return 4  # Pre-master's selection
    elif target_year == 4:
        return 7  # Post-master's, pre/post-exchange
    elif target_year >= 5:
        return 9  # Year 5+
    
    return (target_year - 1) * 2 + 1

def generate_fake_choices_for_gap(choices_needed, year, program):
    """Generate realistic fake choices to fill the gap"""
    
    fake_choice_templates = {
        1: [
            "Social Growth: Attended Chalmers welcome week events",
            "Academic Focus: Joined study groups for challenging courses",
        ],
        2: [
            "Technical Skills: Completed introductory programming projects",
            "Social Growth: Became active in student section activities",
        ],
        3: [
            "Academic Focus: Excelled in advanced engineering courses", 
            "Career Prep: Started exploring internship opportunities",
        ],
        4: [
            "Technical Skills: Led group project in specialized area",
            "Research Focus: Began working with research groups",
        ],
        5: [
            "Career Prep: Secured industry connections",
            "Research Focus: Made progress on thesis project",
        ]
    }
    
    # Get appropriate templates for the year
    templates = fake_choice_templates.get(year, fake_choice_templates[3])
    
    # Cycle through templates if we need more choices than templates
    fake_choices = []
    for i in range(choices_needed):
        template = templates[i % len(templates)]
        
        # Add some variation
        if i >= len(templates):
            template = template.replace(":", f" (Advanced):", 1)
        
        fake_choices.append(template)
    
    return fake_choices

def debug_jump_to_state_internal(session_id, year, choices):
    """Internal function to jump to a specific game state with proper choice history"""
    try:
        # Create or get the session
        if session_id not in game_sessions:
            game_sessions[session_id] = PlayerProfile()
            game_sessions[session_id].name = f"Debug User {session_id}"
            game_sessions[session_id].program = "TKDAT"  # Default program
        
        profile = game_sessions[session_id]
        
        # Reset the profile to clean state
        profile.life_choices = []
        profile.scenario_types_encountered = set()
        profile.comprehensive_history = ""
        profile.personality_summary = ""
        profile.current_situation = f"starting their studies in year {year}"
        profile.year = year
        
        # Calculate target choices needed for this year
        target_choices = calculate_choices_for_year(year)
        
        # If user provided choices, use them
        provided_choices = [choice.strip() for choice in choices if choice.strip()]
        
        # If we need more choices than provided, generate fake ones
        if len(provided_choices) < target_choices:
            fake_choices = generate_fake_choices_for_gap(
                target_choices - len(provided_choices), 
                year, 
                profile.program
            )
            all_choices = provided_choices + fake_choices
        else:
            # Use only the first target_choices if too many provided
            all_choices = provided_choices[:target_choices]
        
        # Add all choices
        for i, choice in enumerate(all_choices):
            if choice.strip():
                # Parse choice format "Type: Description"
                if ':' in choice:
                    choice_type, description = choice.split(':', 1)
                    choice_type = choice_type.strip()
                    description = description.strip()
                else:
                    choice_type = "General"
                    description = choice.strip()
                
                # Add choice with fake context
                choice_context = profile.add_choice(
                    choice_description=description,
                    choice_type=choice_type,
                    choice_data={
                        'situation': f"During your studies, you decided to focus on {choice_type.lower()}.",
                        'character_implication': f"This choice helped shape your character through {description.lower()}",
                        'options': []
                    }
                )
        
        # Force set the year and choices count
        profile.year = year
        
        # Update master's program if we're in year 4+
        if year >= 4 and not profile.selected_masters_program:
            profile.selected_masters_program = "Computer Science and Engineering (Master's)"
        
        # Update profile summaries if LLM is available
        if llm and all_choices:
            try:
                profile_service = ProfileUpdateService(llm)
                # Use the last choice for summary update
                last_choice = all_choices[-1] if all_choices else "General: Started studies"
                if ':' in last_choice:
                    choice_type, description = last_choice.split(':', 1)
                else:
                    choice_type, description = "General", last_choice
                
                fake_context = {
                    'choice_description': description.strip(),
                    'choice_type': choice_type.strip(),
                    'situation': f"Jumping to year {year} state",
                    'character_implication': f"Represents cumulative growth through {len(all_choices)} choices",
                    'other_options': []
                }
                
                profile_service.update_profile_summaries(profile, fake_context)
            except Exception as e:
                logger.warning(f"Could not update profile summaries: {e}")
        
        logger.info(f"🐛 DEBUG: Jumped session {session_id} to year {year} with {len(all_choices)} choices")
        
        return jsonify({
            "success": True,
            "message": f"Jumped to Year {year} with {len(all_choices)} choices",
            "profile": profile.to_dict(),
            "session_id": session_id,
            "debug_info": {
                "target_choices": target_choices,
                "provided_choices": len(provided_choices),
                "fake_choices_added": len(all_choices) - len(provided_choices),
                "choices_breakdown": all_choices
            }
        })
        
    except Exception as e:
        logger.error(f"Error in jump_to_state_internal: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/debug/scenario_triggers', methods=['GET'])
def debug_scenario_triggers():
    """Get information about when different scenarios trigger"""
    try:
        return jsonify({
            "success": True,
            "scenario_triggers": {
                "masters_selection": {
                    "condition": "choices_count == 5 AND not profile.selected_masters_program",
                    "description": "Triggers at exactly 5 choices during Year 3 transition",
                    "debug_setup": "trigger_masters"
                },
                "exchange_opportunity": {
                    "condition": "choices_count in [6, 8] AND profile.year >= 4 AND not _has_done_exchange() AND 50% chance",
                    "description": "50% chance to trigger at 6 or 8 choices during master's years",
                    "debug_setups": ["trigger_exchange_6", "trigger_exchange_8"]
                },
                "thesis_selection": {
                    "condition": "choices_count == 9 AND not _has_done_thesis()",
                    "description": "Triggers at exactly 9 choices during Year 5",
                    "debug_setup": "trigger_thesis"
                },
                "career_preparation": {
                    "condition": "choices_count == 10",
                    "description": "Triggers at exactly 10 choices during late Year 5",
                    "debug_setup": "trigger_career"
                },
                "graduation": {
                    "condition": "choices_count >= 11",
                    "description": "Triggers at 11+ choices for graduation celebration",
                    "debug_setup": "trigger_graduation"
                },
                "ai_generated": {
                    "condition": "All other cases",
                    "description": "Default AI-generated scenarios for regular choices",
                    "debug_setup": "Any setup with different choice counts"
                }
            },
            "year_progression": {
                "year_1": "0-1 choices",
                "year_2": "2-3 choices", 
                "year_3": "4-5 choices (master's selection at 5)",
                "year_4": "6-7 choices (exchange possible at 6 or 8)",
                "year_5": "8+ choices (thesis at 9, career at 10, graduation at 11+)"
            }
        })
    except Exception as e:
        logger.error(f"Error getting scenario triggers: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# ==================== MAIN ====================

if __name__ == '__main__':
    logger.info("🚀 Starting Chalmers Life Journey with SIMPLIFIED AI Director...")
    logger.info("📁 Validating knowledge files...")
    
    # Validate knowledge files exist - updated for new system
    required_files = [
        'sports.txt', 'studies.txt',  # New text files
        'courses_in_program.json', 'course_summary_simplified.json'  # New JSON files
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
    
    # Show registered tools
    logger.info("🔧 Registered tools:")
    for tool_name in _tool_registry.keys():
        logger.info(f"  - {tool_name}")
    
    logger.info("🎬 AI Director initialized with simplified tools:")
    if director:
        for tool_name, tool in director.tools.items():
            description = tool.__class__.__doc__ or f"Tool for {tool_name}"
            logger.info(f"  - {tool_name}: {description.strip()[:60]}{'...' if len(description.strip()) > 60 else ''}")
        logger.info(f"🎬 Total tools available: {len(director.tools)}")
    
    logger.info("🌐 Server starting on http://localhost:5000")
    logger.info("🧪 Test tools at: http://localhost:5000/api/test_tools")
    app.run(debug=True, host='0.0.0.0', port=5000)