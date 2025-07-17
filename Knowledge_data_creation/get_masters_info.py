"""
Programme Info Extractor

This script extracts basic information about both master's and bachelor's programmes 
and generates AI-powered explanations tailored for pre-bachelor students.

Requirements:
- Input file: programme syllabuses JSON in 'Files' directory
- Python packages: python-dotenv, litellm
- Environment variables: LITELLM_API_KEY, LITELLM_BASE_URL

Usage:
    python programme_info_extractor.py
"""

import os
import json
import pprint
import time
from dotenv import load_dotenv
import litellm
from tqdm import tqdm
from typing import Optional, List, Dict, Any


class ProgrammeInfoExtractor:
    """
    A class to extract basic information about both master's and bachelor's programmes.
    """
    
    def __init__(self, data_file: str = None, api_key: str = None, base_url: str = None, model: str = None):
        """
        Initialize the programme info extractor.
        
        Args:
            data_file: Path to the programme data JSON file. If None, uses default location.
            api_key: API key for LLM service (defaults to environment variable)
            base_url: Base URL for LLM service (defaults to environment variable)
            model: LLM model to use (defaults to gpt-4.1-2025-04-14)
        """
        # Load environment variables
        load_dotenv()
        
        # LLM setup
        self.api_key = api_key or os.getenv('LITELLM_API_KEY')
        self.base_url = base_url or os.getenv('LITELLM_BASE_URL', 'https://anast.ita.chalmers.se:4000')
        self.model = model or "gpt-4.1-2025-04-14"
        
        # Configure litellm
        litellm.api_base = self.base_url
        
        # Load programme data
        self.data_file = data_file or os.path.join("Files", "pew_programmesyllabuses_en-20250623_061305.json")
        self.programmes_data = self._load_programme_data()
        self.programmes = [hit["_source"] for hit in self.programmes_data["hits"]["hits"]]
        
        if not self.api_key:
            print("⚠️  Warning: No API key found. AI explanation generation will not work.")
    
    def _load_programme_data(self) -> Dict[str, Any]:
        """
        Load programme data from JSON file.
        
        Returns:
            Dictionary containing programme data
            
        Raises:
            FileNotFoundError: If programme data file doesn't exist
        """
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(
                f"Programme data file not found: {self.data_file}\n"
                "Please ensure the programme data file is in the 'Files' directory."
            )
        
        with open(self.data_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def get_all_programmes_by_type(self, programme_type: str, year: str = "2024/2025") -> List[Dict[str, str]]:
        """
        Get all programmes of a specific type with their basic information.
        
        Args:
            programme_type: Type of programme ("master" or "bachelor")
            year: Academic year to filter by
            
        Returns:
            List of dictionaries with code, name, and explanation
        """
        programmes_list = []
        
        for programme in self.programmes:
            # Skip fake programmes
            if programme.get("isFake") == 1:
                continue
            
            # Filter by academic year
            if programme.get("acYear") != year:
                continue
            
            # Check programme type based on code prefix
            code = programme.get("pCode", "").strip()
            if programme_type == "master" and not code.startswith("MP"):
                continue
            elif programme_type == "bachelor" and not code.startswith("TK"):
                continue
            
            # Extract basic information
            name = self._clean_name(programme.get("name", ""))
            explanation = self._extract_explanation(programme)
            
            programme_info = {
                "code": code,
                "name": name,
                "explanation": explanation
            }
            
            programmes_list.append(programme_info)
        
        # Sort by code for consistent output
        return sorted(programmes_list, key=lambda x: x["code"])
    
    def generate_explanation_for_prebach_students(self, programme_data: Dict[str, Any], programme_type: str = "master") -> str:
        """
        Generate an AI explanation of a programme for pre-bachelor students.
        
        Args:
            programme_data: Complete programme data dictionary
            programme_type: Type of programme ("master" or "bachelor")
            
        Returns:
            AI-generated explanation tailored for pre-bachelor students
            
        Raises:
            ValueError: If programme_data is empty
            RuntimeError: If LLM call fails
        """
        if not programme_data:
            raise ValueError("Missing programme_data")

        # Format the programme data for the LLM
        programme_info = pprint.pformat(programme_data, width=120, compact=True)
        
        if programme_type == "master":
            prompt = (
                "You are writing programme descriptions for pre-bachelor students who are considering their future master's degree options. "
                "Based on the following master's programme information, write a compelling and informative explanation that helps "
                "pre-bachelor students understand what this programme is about, what they'll learn, and what career opportunities it opens up. "
                "Write in an engaging tone that speaks directly to students. Keep it concise (3-4 sentences) but inspiring. "
                "Focus on practical applications, real-world impact, and the exciting possibilities this programme offers. "
                "Avoid mentioning the programme name directly since it will be shown separately.\n\n"
                f"Programme information:\n{programme_info}"
            )
        else:  # bachelor
            prompt = (
                "You are writing programme descriptions for pre-bachelor students who are considering their bachelor's degree options. "
                "Based on the following bachelor's programme information, write a compelling and informative explanation that helps "
                "students understand what this programme is about, what they'll learn, and what career paths it can lead to. "
                "Write in an engaging tone that speaks directly to students. Keep it concise (3-4 sentences) but inspiring. "
                "Focus on practical applications, real-world impact, and the exciting possibilities this programme offers. "
                "Avoid mentioning the programme name directly since it will be shown separately.\n\n"
                f"Programme information:\n{programme_info}"
            )

        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                api_key=self.api_key,
                max_tokens=300,
                temperature=0.7,
                base_url=self.base_url
            )

            explanation = response.choices[0].message.content.strip()
            return explanation

        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}")

    def get_specified_programmes_with_ai(
        self, 
        programme_codes: List[str], 
        programme_type: str,
        year: str = "2024/2025",
        dry_run: bool = False
    ) -> List[Dict[str, str]]:
        """
        Get information for specified programmes with AI-generated explanations.
        
        Args:
            programme_codes: List of programme codes to extract
            programme_type: Type of programme ("master" or "bachelor")
            year: Academic year to filter by
            dry_run: If True, uses placeholder explanations instead of calling LLM
            
        Returns:
            List of dictionaries with code, name, and AI-generated explanation
        """
        programmes_list = []
        found_codes = set()
        
        print(f"\n{'='*60}")
        print(f"EXTRACTING {len(programme_codes)} {programme_type.upper()}'S PROGRAMMES WITH AI EXPLANATIONS")
        print(f"{'='*60}")
        
        mode = "dry run" if dry_run else "AI explanation generation"
        print(f"Mode: {mode}\n")
        
        # Find programmes and generate explanations
        for programme in tqdm(self.programmes, desc="Processing programmes", unit="programme"):
            # Skip fake programmes
            if programme.get("isFake") == 1:
                continue
            
            # Filter by academic year
            if programme.get("acYear") != year:
                continue
            
            # Check if this programme is in our specified list
            code = programme.get("pCode", "").strip()
            if code not in programme_codes:
                continue
            
            found_codes.add(code)
            
            # Extract basic information
            name = self._clean_name(programme.get("name", ""))
            
            # Generate explanation
            if dry_run:
                explanation = "(AI explanation placeholder)"
            else:
                try:
                    explanation = self.generate_explanation_for_prebach_students(programme, programme_type)
                except Exception as e:
                    explanation = f"Error generating explanation: {e}"
                    print(f"  ⚠️ Error for {code}: {e}")
            
            programme_info = {
                "code": code,
                "name": name,
                "explanation": explanation
            }
            
            programmes_list.append(programme_info)
        
        # Report missing programmes
        missing_codes = set(programme_codes) - found_codes
        if missing_codes:
            print(f"\n⚠️ Warning: Could not find programmes: {sorted(missing_codes)}")
        
        print(f"\n✅ Successfully processed {len(programmes_list)} programmes")
        
        # Sort by the order in the input list for consistent output
        programmes_list.sort(key=lambda x: programme_codes.index(x["code"]) if x["code"] in programme_codes else 999)
        
        return programmes_list
    
    def _clean_name(self, name: str) -> str:
        """
        Clean and standardize programme names.
        
        Args:
            name: Raw programme name
            
        Returns:
            Cleaned programme name
        """
        if not name:
            return ""
        
        name = name.strip()
        
        # Remove common suffixes and clean up
        if name.endswith(", MSc Programme"):
            name = name.replace(", MSc Programme", "")
        elif name.endswith(" MSc Programme"):
            name = name.replace(" MSc Programme", "")
        elif name.endswith(", MSc"):
            name = name.replace(", MSc", "")
        elif name.endswith(", BSc Programme"):
            name = name.replace(", BSc Programme", "")
        elif name.endswith(" BSc Programme"):
            name = name.replace(" BSc Programme", "")
        elif name.endswith(", BSc"):
            name = name.replace(", BSc", "")
        
        # Additional cleaning
        name = name.replace("Msc", "MSc")
        name = name.replace("Bsc", "BSc")
        name = name.replace(" And ", " and ")
        
        return name.strip()
    
    def _extract_explanation(self, programme: Dict[str, Any]) -> str:
        """
        Extract or create an explanation for the programme.
        
        Args:
            programme: Programme data dictionary
            
        Returns:
            Programme explanation/description
        """
        # Try different fields that might contain programme description
        explanation_fields = [
            "description",
            "programmeDescription", 
            "content",
            "aim",
            "programmeAim",
            "generalInfo",
            "info"
        ]
        
        for field in explanation_fields:
            if field in programme and programme[field]:
                explanation = programme[field].strip()
                if explanation:
                    return self._clean_explanation(explanation)
        
        # If no explanation found, create a basic one based on the name
        name = programme.get("name", "")
        code = programme.get("pCode", "")
        
        if code.startswith("MP"):
            return f"Master's programme in {self._clean_name(name).lower()}."
        elif code.startswith("TK"):
            return f"Bachelor's programme in {self._clean_name(name).lower()}."
        else:
            return f"Programme in {self._clean_name(name).lower()}."
    
    def _clean_explanation(self, explanation: str) -> str:
        """
        Clean up the explanation text.
        
        Args:
            explanation: Raw explanation text
            
        Returns:
            Cleaned explanation
        """
        if not explanation:
            return ""
        
        # Basic cleaning
        explanation = explanation.strip()
        
        # Remove excessive whitespace
        explanation = " ".join(explanation.split())
        
        # Ensure it ends with a period
        if explanation and not explanation.endswith(('.', '!', '?')):
            explanation += "."
        
        return explanation
    
    def create_programmes_json_with_ai(
        self, 
        programme_codes: List[str], 
        programme_type: str,
        year: str = "2024/2025",
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Create the JSON structure for programmes with AI-generated explanations.
        
        Args:
            programme_codes: List of specific programme codes to include
            programme_type: Type of programme ("master" or "bachelor")
            year: Academic year to filter by
            dry_run: If True, uses placeholder explanations instead of calling LLM
            
        Returns:
            Dictionary in the format {"programs": [...]}
        """
        programmes_list = self.get_specified_programmes_with_ai(programme_codes, programme_type, year, dry_run)
        return {"programs": programmes_list}

    def save_programmes_json_with_ai(
        self, 
        output_path: str, 
        programme_codes: List[str], 
        programme_type: str,
        year: str = "2024/2025",
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Save programmes information with AI explanations to JSON file.
        
        Args:
            output_path: Path where to save the JSON file
            programme_codes: List of specific programme codes to include
            programme_type: Type of programme ("master" or "bachelor")
            year: Academic year to filter by
            dry_run: If True, uses placeholder explanations instead of calling LLM
            
        Returns:
            The generated programmes data
        """
        start_time = time.time()
        
        # Create the JSON data with AI explanations
        programmes_data = self.create_programmes_json_with_ai(programme_codes, programme_type, year, dry_run)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(programmes_data, f, indent=2, ensure_ascii=False)
        
        total_time = time.time() - start_time
        print(f"\n✓ Saved {programme_type}'s programmes data to: {output_path}")
        print(f"⏱ Total time: {total_time / 60:.2f} minutes")
        
        return programmes_data
    
    def preview_programmes(self, programmes_data: Dict[str, Any], max_programs: int = 5) -> None:
        """
        Preview the extracted programmes data.
        
        Args:
            programmes_data: The programmes data dictionary
            max_programs: Maximum number of programmes to preview
        """
        programs = programmes_data.get("programs", [])
        
        print(f"\n{'='*60}")
        print(f"PREVIEW OF EXTRACTED PROGRAMMES ({len(programs)} total)")
        print(f"{'='*60}")
        
        for i, program in enumerate(programs[:max_programs]):
            print(f"\nProgram {i+1}:")
            print(f"  Code: {program['code']}")
            print(f"  Name: {program['name']}")
            print(f"  Explanation: {program['explanation'][:100]}{'...' if len(program['explanation']) > 100 else ''}")
        
        if len(programs) > max_programs:
            print(f"\n... and {len(programs) - max_programs} more programmes")
    
    def get_available_programme_codes(self, programme_type: str, year: str = "2024/2025") -> List[str]:
        """
        Get list of available programme codes by type.
        
        Args:
            programme_type: Type of programme ("master" or "bachelor")
            year: Academic year to filter by
            
        Returns:
            List of available programme codes
        """
        programme_codes = []
        
        for programme in self.programmes:
            # Skip fake programmes
            if programme.get("isFake") == 1:
                continue
            
            # Filter by academic year
            if programme.get("acYear") != year:
                continue
            
            # Check programme type
            code = programme.get("pCode", "").strip()
            if programme_type == "master" and code.startswith("MP"):
                programme_codes.append(code)
            elif programme_type == "bachelor" and code.startswith("TK"):
                programme_codes.append(code)
        
        return sorted(list(set(programme_codes)))
    
    def get_all_masters_codes(self) -> List[str]:
        """
        Get the complete list of all master's programme codes.
        
        Returns:
            List of all master's programme codes
        """
        return [
            "MPAEM",  # Materials Engineering
            "MPALG",  # Computer Science - Algorithms, Languages and Logic
            "MPAME",  # Applied Mechanics
            "MPARC",  # Architecture and Urban Design
            "MPBDP",  # Entrepreneurship and Business Design
            "MPBIO",  # Biotechnology
            "MPCAS",  # Complex Adaptive Systems
            "MPCSN",  # Computer Systems and Networks
            "MPDCM",  # Design and Construction Project Management
            "MPDES",  # Industrial Design Engineering
            "MPDSC",  # Data Science and AI
            "MPDSD",  # Architecture and Planning Beyond Sustainability
            "MPEES",  # Embedded Electronic System Design
            "MPENM",  # Engineering Mathematics and Computational Science
            "MPEPO",  # Sustainable Electric Power Engineering and Electromobility
            "MPHPC",  # High-Performance Computer Systems
            "MPICT",  # Information and Communication Technology
            "MPIDE",  # Interaction Design and Technologies
            "MPIEE",  # Infrastructure and Environmental Engineering
            "MPISC",  # Innovative and Sustainable Chemical Engineering
            "MPLOL",  # Learning and Leadership
            "MPMAR",  # Maritime Management
            "MPMCN",  # Materials Chemistry
            "MPMED",  # Biomedical Engineering
            "MPMEI",  # Management and Economics Of Innovation
            "MPMOB",  # Mobility Engineering
            "MPNAT",  # Nanotechnology
            "MPPDE",  # Product Development
            "MPPEN",  # Production Engineering
            "MPPHS",  # Physics
            "MPQOM",  # Quality and Operations Management
            "MPSCM",  # Supply Chain Management
            "MPSEB",  # Structural Engineering and Building Technology
            "MPSES",  # Sustainable Energy Systems
            "MPSOF",  # Software Engineering and Technology
            "MPSOV",  # Sound and Vibration
            "MPSYS",  # Systems, Control and Mechatronics
            "MPTSE",  # Industrial Ecology
            "MPWPS",  # Wireless, Photonics and Space Engineering
        ]

    def get_all_bachelors_codes(self) -> List[str]:
        """
        Get the complete list of all bachelor's programme codes.
        
        Returns:
            List of all bachelor's programme codes
        """
        return [
            "TKATK",  # Architecture and Engineering
            "TKAUT",  # Automation and Mechatronics Engineering
            "TKBIO",  # Bioengineering
            "TKDAT",  # Computer Science and Engineering
            "TKDES",  # Industrial Design Engineering
            "TKELT",  # Electrical Engineering
            "TKGBS",  # Global Systems Engineering
            "TKIEK",  # Industrial Engineering and Management
            "TKITE",  # Software Engineering
            "TKKEF",  # Chemical Engineering With Engineering Physics
            "TKKMT",  # Chemical Engineering
            "TKMAS",  # Mechanical Engineering
            "TKMED",  # Biomedical Engineering
            "TKSAM",  # Civil Engineering
            "TKTEM",  # Engineering Mathematics
            "TKTFY",  # Engineering Physics
            "TKARK",  # Architecture
        ]


def main():
    """
    Main function to extract both master's and bachelor's programmes with AI-generated explanations.
    """
    try:
        # Configuration
        TARGET_YEAR = "2024/2025"
        DRY_RUN = False  # Set to True to test without calling LLM
        
        # Initialize the extractor
        extractor = ProgrammeInfoExtractor()
        
        print(f"Programme Info Extractor with AI Explanations")
        print(f"Target Year: {TARGET_YEAR}")
        print(f"Dry Run Mode: {DRY_RUN}")
        
        # Get the programme lists
        all_masters_codes = extractor.get_all_masters_codes()
        all_bachelors_codes = extractor.get_all_bachelors_codes()
        
        print(f"\n{'='*60}")
        print(f"EXTRACTING PROGRAMMES")
        print(f"{'='*60}")
        print(f"Master's programmes: {len(all_masters_codes)}")
        print(f"Bachelor's programmes: {len(all_bachelors_codes)}")
        
        # Extract master's programmes with AI explanations
        print(f"\n🎓 Processing Master's Programmes...")
        masters_data = extractor.save_programmes_json_with_ai(
            output_path=os.path.join("Files_created", "masters_programs.json"),
            programme_codes=all_masters_codes,
            programme_type="master",
            year=TARGET_YEAR,
            dry_run=DRY_RUN
        )
        
        # Extract bachelor's programmes with AI explanations
        print(f"\n🎓 Processing Bachelor's Programmes...")
        bachelors_data = extractor.save_programmes_json_with_ai(
            output_path=os.path.join("Files_created", "programs.json"),
            programme_codes=all_bachelors_codes,
            programme_type="bachelor",
            year=TARGET_YEAR,
            dry_run=DRY_RUN
        )
        
        # Preview the results
        print(f"\n{'='*60}")
        print("MASTER'S PROGRAMMES PREVIEW")
        print(f"{'='*60}")
        extractor.preview_programmes(masters_data, max_programs=2)
        
        print(f"\n{'='*60}")
        print("BACHELOR'S PROGRAMMES PREVIEW")
        print(f"{'='*60}")
        extractor.preview_programmes(bachelors_data, max_programs=2)
        
        print(f"\n{'='*60}")
        print("EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"📄 Master's programmes file: Files_created/masters_programs.json")
        print(f"📄 Bachelor's programmes file: Files_created/programs.json")
        print(f"📊 Total master's programmes extracted: {len(masters_data['programs'])}")
        print(f"📊 Total bachelor's programmes extracted: {len(bachelors_data['programs'])}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())