"""
Programme Analyzer

This script analyzes university programme data to:
1. Categorize programmes by degree type
2. Filter and group programmes by academic year
3. Extract accreditation information
4. Generate clean, organized reports

Requirements:
- Input file: programme syllabuses JSON in 'Files' directory

Usage:
    python programme_analyzer.py
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any


class ProgrammeAnalyzer:
    """
    A class to analyze university programme data and generate reports.
    """
    
    def __init__(self, data_file: str = None):
        """
        Initialize the programme analyzer.
        
        Args:
            data_file: Path to the programme data JSON file. If None, uses default location.
        """
        self.data_file = data_file or os.path.join("Files", "pew_programmesyllabuses_en-20250623_061305.json")
        self.programmes_data = self._load_programme_data()
        self.programmes = [hit["_source"] for hit in self.programmes_data["hits"]["hits"]]
    
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
    
    def normalize_degree(self, degree_text: str) -> str:
        """
        Normalize degree types into clean categories.
        
        Args:
            degree_text: Raw degree text from the data
            
        Returns:
            Normalized degree category
        """
        if not degree_text or degree_text.strip() == "":
            return "⚠️ Unspecified Degree"
        
        degree = degree_text.lower()
        
        # Define degree mappings in order of specificity
        degree_mappings = [
            ("master of science in engineering", "Master of Science in Engineering"),
            ("master of science", "Master of Science"),
            ("master of architecture", "Master of Architecture"),
            ("bachelor of science in engineering", "Bachelor of Science in Engineering"),
            ("bachelor of science in nautical science", "Bachelor of Science in Nautical Science"),
            ("bachelor of science in marine engineering", "Bachelor of Science in Marine Engineering"),
            ("bachelor of science in secondary education", "Bachelor of Science in Secondary Education"),
            ("bachelor of science", "Bachelor of Science"),
        ]
        
        for search_term, normalized_name in degree_mappings:
            if search_term in degree:
                return normalized_name
        
        return "Other"
    
    def clean_name(self, name: str) -> str:
        """
        Clean and standardize programme names.
        
        Args:
            name: Raw programme name
            
        Returns:
            Cleaned programme name
        """
        if not name:
            return ""
        
        name = name.strip().title()
        
        # Apply specific cleaning rules
        name = name.replace("Msc Progr", "MSc Programme")
        name = name.replace(" And ", " and ")
        
        return name
    
    def filter_and_group_by_degree(self, year: str) -> Dict[str, List[Tuple[str, str]]]:
        """
        Filter programmes by year and group them by degree type.
        
        Args:
            year: Academic year to filter by (e.g., "2024/2025")
            
        Returns:
            Dictionary mapping degree types to lists of (code, name) tuples
        """
        grouped = defaultdict(list)
        
        for programme in self.programmes:
            # Skip fake programmes
            if programme.get("isFake") == 1:
                continue
            
            # Filter by academic year
            if programme.get("acYear") != year:
                continue
            
            # Categorize and clean data
            degree_group = self.normalize_degree(programme.get("degrees", ""))
            code = programme.get("pCode", "").strip()
            name = self.clean_name(programme.get("name", ""))
            
            grouped[degree_group].append((code, name))
        
        return grouped
    
    def print_grouped_programmes(self, grouped: Dict[str, List[Tuple[str, str]]]) -> None:
        """
        Print programmes grouped by degree type in a formatted way.
        
        Args:
            grouped: Dictionary of degree types and their programmes
        """
        print(f"\n{'='*60}")
        print("PROGRAMMES BY DEGREE TYPE")
        print(f"{'='*60}")
        
        for group in sorted(grouped):
            print(f"\n🎓 Degree Type: {group}")
            
            # Remove duplicates while preserving order
            seen = set()
            unique_programmes = []
            
            for code, name in sorted(grouped[group]):
                if (code, name) not in seen:
                    unique_programmes.append((code, name))
                    seen.add((code, name))
            
            # Print programmes
            for code, name in unique_programmes:
                print(f"  - {code}: {name}")
            
            print(f"  📊 Total: {len(unique_programmes)} programmes")
    
    def find_programme_by_code(self, code: str, year: str) -> Optional[Dict[str, Any]]:
        """
        Find a programme by its code and academic year.
        
        Args:
            code: Programme code to search for
            year: Academic year
            
        Returns:
            Programme data dictionary or None if not found
        """
        matches = [
            p for p in self.programmes
            if p.get("isFake") != 1
            and p.get("pCode", "").strip().upper() == code.upper()
            and p.get("acYear") == year
        ]
        
        return matches[0] if matches else None
    
    def print_programme_json(self, code: str, year: str) -> None:
        """
        Print the complete JSON data for a specific programme.
        
        Args:
            code: Programme code to search for
            year: Academic year
        """
        programme = self.find_programme_by_code(code, year)
        
        if not programme:
            print(f"❌ No programme found with code '{code}' for academic year '{year}'")
            return
        
        print(f"\n📦 Programme '{code}' for year {year}:")
        print(json.dumps(programme, indent=2, ensure_ascii=False))
    
    def get_accredited_masters(self, code: str, year: str) -> List[str]:
        """
        Get the list of accredited master's programmes for a given programme code.
        
        Args:
            code: Programme code to search for
            year: Academic year
            
        Returns:
            List of accredited master's programme names
        """
        programme = self.find_programme_by_code(code, year)
        
        if not programme:
            print(f"❌ No matching programme for code '{code}' in year '{year}'")
            return []
        
        accreditations = programme.get("accreditations", {}).get("forMastersProgramme", [])
        
        if not accreditations:
            print(f"ℹ️ No accredited master's programmes found for '{code}' in {year}")
            return []
        
        print(f"✅ Accredited master's programmes for '{code}' in {year}:")
        for name in accreditations:
            print(f"  - {name}")
        
        return accreditations
    
    def generate_summary_report(self, year: str) -> None:
        """
        Generate a comprehensive summary report for all programmes in a given year.
        
        Args:
            year: Academic year to analyze
        """
        grouped = self.filter_and_group_by_degree(year)
        
        print(f"\n{'='*60}")
        print(f"SUMMARY REPORT FOR ACADEMIC YEAR {year}")
        print(f"{'='*60}")
        
        total_programmes = sum(len(programmes) for programmes in grouped.values())
        total_unique = sum(len(set(programmes)) for programmes in grouped.values())
        
        print(f"📊 Total programmes: {total_programmes}")
        print(f"📊 Unique programmes: {total_unique}")
        print(f"📊 Degree categories: {len(grouped)}")
        
        # Print detailed breakdown
        self.print_grouped_programmes(grouped)
        
        return grouped
    
    def search_programmes(self, search_term: str, year: str) -> List[Tuple[str, str]]:
        """
        Search for programmes containing a specific term.
        
        Args:
            search_term: Term to search for in programme names
            year: Academic year to filter by
            
        Returns:
            List of (code, name) tuples matching the search term
        """
        matches = []
        
        for programme in self.programmes:
            if programme.get("isFake") == 1:
                continue
            
            if programme.get("acYear") != year:
                continue
            
            name = programme.get("name", "").lower()
            if search_term.lower() in name:
                code = programme.get("pCode", "").strip()
                clean_name = self.clean_name(programme.get("name", ""))
                matches.append((code, clean_name))
        
        return matches
    
    def print_dataset_stats(self) -> None:
        """Print basic statistics about the loaded dataset."""
        total_hits = self.programmes_data.get("hits", {}).get("total", {}).get("value", 0)
        returned_docs = len(self.programmes_data.get("hits", {}).get("hits", []))
        
        # Count real vs fake programmes
        real_programmes = sum(1 for p in self.programmes if p.get("isFake") != 1)
        fake_programmes = len(self.programmes) - real_programmes
        
        print(f"Dataset Statistics:")
        print(f"  Total hits: {total_hits}")
        print(f"  Returned documents: {returned_docs}")
        print(f"  Real programmes: {real_programmes}")
        print(f"  Fake programmes: {fake_programmes}")
    
    def get_accredited_masters_codes(self, code: str, year: str) -> List[str]:
        """
        Get the list of accredited master's programme codes for a given programme code.
        
        Args:
            code: Programme code to search for
            year: Academic year
            
        Returns:
            List of accredited master's programme codes
        """
        programme = self.find_programme_by_code(code, year)
        
        if not programme:
            return []
        
        accreditation_data = programme.get("accreditations", {}).get("forMastersProgramme", [])
        
        if not accreditation_data:
            return []
        
        # Extract master codes from the accreditation data
        accredited_codes = []
        for acc_item in accreditation_data:
            if isinstance(acc_item, dict) and 'masterCode' in acc_item:
                master_code = acc_item['masterCode']
                if master_code and master_code not in accredited_codes:
                    accredited_codes.append(master_code)
        
        return sorted(accredited_codes)
    
    def _find_programme_code_by_name_fuzzy(self, name: str, year: str) -> Optional[str]:
        """
        Find a programme code by its name using fuzzy matching.
        
        Args:
            name: Programme name to search for
            year: Academic year
            
        Returns:
            Programme code or None if not found
        """
        if not name or not isinstance(name, str):
            return None
            
        name_lower = name.lower().strip()
        
        for programme in self.programmes:
            if programme.get("isFake") == 1:
                continue
            
            if programme.get("acYear") != year:
                continue
            
            # Check if the programme name matches
            prog_name = programme.get("name", "")
            if not prog_name or not isinstance(prog_name, str):
                continue
                
            prog_name_lower = prog_name.strip().lower()
            
            # Try various matching strategies
            if (prog_name_lower == name_lower or 
                prog_name_lower in name_lower or 
                name_lower in prog_name_lower):
                return programme.get("pCode", "").strip()
        
        return None
    
    def debug_programme_accreditations(self, code: str, year: str) -> None:
        """
        Debug function to show what accreditations exist for a programme.
        
        Args:
            code: Programme code to debug
            year: Academic year
        """
        programme = self.find_programme_by_code(code, year)
        
        if not programme:
            print(f"❌ Programme {code} not found")
            return
        
        print(f"\n🔍 Debug info for {code}:")
        print(f"  Programme name: {programme.get('name', 'N/A')}")
        
        accreditations = programme.get("accreditations", {})
        print(f"  Accreditations structure: {type(accreditations)}")
        
        if isinstance(accreditations, dict):
            print(f"  Accreditation keys: {list(accreditations.keys())}")
            
            masters_acc = accreditations.get("forMastersProgramme", [])
            print(f"  Masters accreditations: {masters_acc}")
            print(f"  Masters accreditations type: {type(masters_acc)}")
            
            if masters_acc:
                print(f"  Sample accreditation names:")
                for i, acc_name in enumerate(masters_acc[:3]):  # Show first 3
                    print(f"    {i+1}. '{acc_name}' (type: {type(acc_name)})")
        else:
            print(f"  Raw accreditations: {accreditations}")
    
    def debug_all_programme_names(self, year: str) -> None:
        """
        Debug function to show all programme names in the dataset.
        
        Args:
            year: Academic year to filter by
        """
        print(f"\n🔍 All programme names for {year}:")
        
        names_and_codes = []
        for programme in self.programmes:
            if programme.get("isFake") == 1:
                continue
            
            if programme.get("acYear") != year:
                continue
            
            name = programme.get("name", "")
            code = programme.get("pCode", "")
            names_and_codes.append((code, name))
        
        # Sort by code for easier reading
        for code, name in sorted(names_and_codes):
            print(f"  {code}: {name}")
        
        print(f"\nTotal programmes: {len(names_and_codes)}")

    
    def _find_programme_code_by_name(self, name: str, year: str) -> Optional[str]:
        """
        Find a programme code by its name and academic year.
        
        Args:
            name: Programme name to search for
            year: Academic year
            
        Returns:
            Programme code or None if not found
        """
        if not name or not isinstance(name, str):
            return None
            
        for programme in self.programmes:
            if programme.get("isFake") == 1:
                continue
            
            if programme.get("acYear") != year:
                continue
            
            # Check if the programme name matches
            prog_name = programme.get("name", "")
            if not prog_name or not isinstance(prog_name, str):
                continue
                
            prog_name = prog_name.strip()
            if prog_name.lower() == name.lower():
                return programme.get("pCode", "").strip()
        
        return None
    
    def extract_accredited_masters_for_programmes(
        self, 
        programme_codes: List[str], 
        year: str
    ) -> Dict[str, List[str]]:
        """
        Extract accredited master's programme codes for multiple programmes.
        
        Args:
            programme_codes: List of programme codes to extract accreditations for
            year: Academic year
            
        Returns:
            Dictionary mapping programme codes to lists of accredited master's codes
        """
        results = {}
        successful_extractions = 0
        
        print(f"\n{'='*60}")
        print(f"EXTRACTING ACCREDITED MASTERS FOR {len(programme_codes)} PROGRAMMES")
        print(f"{'='*60}")
        
        for prog_code in programme_codes:
            accredited_codes = self.get_accredited_masters_codes(prog_code, year)
            results[prog_code] = accredited_codes
            
            if accredited_codes:
                print(f"✓ {prog_code}: {len(accredited_codes)} accredited masters")
                successful_extractions += 1
            else:
                print(f"○ {prog_code}: No accredited masters found")
        
        print(f"\n📊 Extraction Summary:")
        print(f"  Programmes with accredited masters: {successful_extractions}")
        print(f"  Programmes without accredited masters: {len(programme_codes) - successful_extractions}")
        print(f"  Total programmes processed: {len(programme_codes)}")
        
        return results
    
    def save_accredited_masters_json(
        self, 
        accredited_data: Dict[str, List[str]], 
        year: str
    ) -> None:
        """
        Save accredited masters data to JSON files.
        
        Args:
            accredited_data: Dictionary of programme codes to accredited masters
            year: Academic year for filename
        """
        # Create output directory
        output_dir = "Files_created"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create comprehensive output with metadata
        all_accredited_codes = set()
        for codes in accredited_data.values():
            all_accredited_codes.update(codes)
        
        comprehensive_data = {
            "programme_accreditations": accredited_data,
            "all_accredited_masters": sorted(list(all_accredited_codes)),
            "summary": {
                "total_programmes": len(accredited_data),
                "programmes_with_accreditations": sum(1 for codes in accredited_data.values() if codes),
                "total_unique_accredited_masters": len(all_accredited_codes),
                "academic_year": year,
                "extraction_timestamp": self._get_timestamp()
            }
        }
        
        comprehensive_output = os.path.join(output_dir, f"accredited_masters.json")
        with open(comprehensive_output, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved comprehensive data to: {comprehensive_output}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_predefined_bachelor_programmes(self) -> List[str]:
        """
        Get the predefined list of bachelor programme codes.
        
        Returns:
            List of bachelor programme codes
        """
        return [
            "TKARK",  # Architecture
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
        ]


def main():
    """
    Main function to demonstrate usage of the ProgrammeAnalyzer.
    
    This function shows how to:
    1. Initialize the analyzer
    2. Generate summary reports
    3. Search for specific programmes
    4. Get accreditation information
    """
    try:
        # Configuration
        TARGET_YEAR = "2024/2025"  # Change this if needed
        
        # Initialize the analyzer
        analyzer = ProgrammeAnalyzer()
        
        # Print dataset information
        analyzer.print_dataset_stats()
        
        # Generate comprehensive summary report
        print(f"\n{'='*60}")
        print("GENERATING COMPREHENSIVE REPORT")
        print(f"{'='*60}")
        
        grouped = analyzer.generate_summary_report(TARGET_YEAR)
        
        # Example searches and queries
        print(f"\n{'='*60}")
        print("EXAMPLE QUERIES")
        print(f"{'='*60}")
        
        # Example 1: Get accredited masters for a specific programme
        print(f"\n🔍 Example 1: Accredited masters for TKDAT")
        analyzer.get_accredited_masters("TKDAT", TARGET_YEAR)
        
        # Example 2: Search for programmes containing "computer"
        print(f"\n🔍 Example 2: Search for programmes containing 'computer'")
        computer_programmes = analyzer.search_programmes("computer", TARGET_YEAR)
        if computer_programmes:
            print("Found programmes:")
            for code, name in computer_programmes:
                print(f"  - {code}: {name}")
        else:
            print("No programmes found containing 'computer'")
        
        # Example 3: Print detailed info for a specific programme (uncomment to use)
        # print(f"\n🔍 Example 3: Detailed info for TKDAT")
        # analyzer.print_programme_json("TKDAT", TARGET_YEAR)
        
        # Example 4: Extract accredited masters for bachelor programmes
        print(f"\n🔍 Example 4: Extract accredited masters for bachelor programmes")
        
        # First, let's debug a few programmes to see what's happening
        print(f"\n🔧 Debug: Let's check what accreditations exist...")
        sample_programmes = ["TKDAT", "TKELT", "TKMAS"]
        for prog in sample_programmes:
            analyzer.debug_programme_accreditations(prog, TARGET_YEAR)
        
        bachelor_programmes = analyzer.get_predefined_bachelor_programmes()
        accredited_data = analyzer.extract_accredited_masters_for_programmes(bachelor_programmes, TARGET_YEAR)
        
        # Save to JSON files
        analyzer.save_accredited_masters_json(accredited_data, TARGET_YEAR)
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())