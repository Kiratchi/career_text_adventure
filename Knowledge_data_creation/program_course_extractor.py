"""
Programme Course Extractor

This script extracts course codes and study periods from university programme data to:
1. Extract courses for individual programmes
2. Map courses to their study periods
3. Generate comprehensive reports for multiple programmes
4. Create structured JSON outputs for further analysis

Requirements:
- Input file: programme study plans JSON in 'Files' directory

Usage:
    python programme_course_extractor.py
"""

import os
import json
from collections import defaultdict
from typing import Optional, List, Dict, Set, Any
from pprint import pprint


class ProgrammeCourseExtractor:
    """
    A class to extract course codes and study periods from programme data.
    """
    
    def __init__(self, data_file: str = None):
        """
        Initialize the programme course extractor.
        
        Args:
            data_file: Path to the programme data JSON file. If None, uses default location.
        """
        self.data_file = data_file or os.path.join("Files", "pew_studyprogrammes_en-20250623_061255.json")
        self.data = self._load_data()
    
    def _load_data(self) -> Dict[str, Any]:
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
    
    def find_programme_by_code(self, code: str, year: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Find a programme by its code and optional academic year.
        
        Args:
            code: Programme code to search for
            year: Academic year (e.g., "2024/2025"). If None, searches all years.
            
        Returns:
            Programme data dictionary or None if not found
        """
        for entry in self.data.get("hits", {}).get("hits", []):
            programme = entry.get("_source", {})
            
            # Skip fake programmes
            if programme.get("isFake") == 1:
                continue
            
            # Check programme code
            if programme.get("programCode", "").upper() != code.upper():
                continue
            
            # Check academic year if specified
            if year and programme.get("acYear") != year:
                continue
            
            return programme
        
        return None
    
    def extract_course_codes_and_periods(
        self, 
        code: str, 
        year: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """
        Extract course codes and their study periods for a given programme code.
        
        Args:
            code: Programme code to extract courses for
            year: Academic year to filter by
            
        Returns:
            Dictionary with course codes as keys and list of study periods as values
            
        Raises:
            ValueError: If programme is not found
        """
        programme = self.find_programme_by_code(code, year)
        
        if not programme:
            raise ValueError(f"Programme '{code}' not found in year '{year}'")
        
        course_periods = defaultdict(set)
        
        # Extract courses from all year classes
        for year_class in programme.get("yearClasses", []):
            for moment in year_class.get("courseMoments", []):
                study_period = moment.get("studyPerodId", "OTH")
                course_code = moment.get("courseCode", "???")
                course_periods[course_code].add(study_period)
        
        # Convert sets to sorted lists for consistent output
        result = {
            course_code: sorted(list(periods)) 
            for course_code, periods in course_periods.items()
        }
        
        return result
    
    def extract_multiple_programmes(
        self, 
        programme_codes: List[str], 
        year: Optional[str] = None
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Extract course codes and study periods for multiple programmes.
        
        Args:
            programme_codes: List of programme codes to extract
            year: Academic year to filter by
            
        Returns:
            Nested dictionary: {programme_code: {course_code: [study_periods]}}
        """
        all_programmes_data = {}
        successful_extractions = 0
        failed_extractions = 0
        
        print(f"\n{'='*60}")
        print(f"EXTRACTING COURSES FOR {len(programme_codes)} PROGRAMMES")
        print(f"{'='*60}")
        
        for programme_code in programme_codes:
            try:
                programme_data = self.extract_course_codes_and_periods(programme_code, year)
                all_programmes_data[programme_code] = programme_data
                print(f"✓ Successfully extracted data for {programme_code} ({len(programme_data)} courses)")
                successful_extractions += 1
            except ValueError as e:
                print(f"✗ Error extracting {programme_code}: {e}")
                all_programmes_data[programme_code] = {}
                failed_extractions += 1
        
        print(f"\n📊 Extraction Summary:")
        print(f"  Successful: {successful_extractions}")
        print(f"  Failed: {failed_extractions}")
        print(f"  Total: {len(programme_codes)}")
        
        return all_programmes_data
    
    def save_to_json(self, data: Dict[str, Any], output_path: str) -> None:
        """
        Save data to a JSON file.
        
        Args:
            data: Data to save
            output_path: Path where to save the JSON file
        """
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Data saved to {output_path}")
    
    def analyze_extracted_data(self, extracted_data: Dict[str, Dict[str, List[str]]]) -> Dict[str, Any]:
        """
        Analyze the extracted data to provide insights.
        
        Args:
            extracted_data: Data extracted from programmes
            
        Returns:
            Dictionary containing analysis results
        """
        # Collect all unique courses
        all_courses = set()
        for programme_data in extracted_data.values():
            all_courses.update(programme_data.keys())
        
        # Count study periods
        study_period_counts = defaultdict(int)
        for programme_data in extracted_data.values():
            for periods in programme_data.values():
                for period in periods:
                    study_period_counts[period] += 1
        
        # Programme statistics
        programme_stats = {}
        for programme_code, courses in extracted_data.items():
            programme_stats[programme_code] = {
                "total_courses": len(courses),
                "unique_periods": len(set(
                    period for periods in courses.values() for period in periods
                ))
            }
        
        analysis = {
            "total_programmes": len(extracted_data),
            "total_unique_courses": len(all_courses),
            "study_period_distribution": dict(study_period_counts),
            "programme_statistics": programme_stats,
            "all_unique_courses": sorted(list(all_courses))
        }
        
        return analysis
    
    def print_analysis_report(self, analysis: Dict[str, Any]) -> None:
        """
        Print a comprehensive analysis report.
        
        Args:
            analysis: Analysis results from analyze_extracted_data
        """
        print(f"\n{'='*60}")
        print("ANALYSIS REPORT")
        print(f"{'='*60}")
        
        print(f"📊 Overall Statistics:")
        print(f"  Total programmes: {analysis['total_programmes']}")
        print(f"  Total unique courses: {analysis['total_unique_courses']}")
        
        print(f"\n📅 Study Period Distribution:")
        for period, count in sorted(analysis['study_period_distribution'].items()):
            print(f"  {period}: {count} course instances")
        
        print(f"\n🎓 Programme Statistics (Top 10 by course count):")
        sorted_programmes = sorted(
            analysis['programme_statistics'].items(),
            key=lambda x: x[1]['total_courses'],
            reverse=True
        )
        
        for programme, stats in sorted_programmes[:10]:
            print(f"  {programme}: {stats['total_courses']} courses, {stats['unique_periods']} periods")
        
        if len(sorted_programmes) > 10:
            print(f"  ... and {len(sorted_programmes) - 10} more programmes")
    
    def preview_extracted_data(self, extracted_data: Dict[str, Dict[str, List[str]]], max_programmes: int = 5) -> None:
        """
        Preview the extracted data for verification.
        
        Args:
            extracted_data: Data extracted from programmes
            max_programmes: Maximum number of programmes to show
        """
        print(f"\n{'='*60}")
        print("PREVIEW OF EXTRACTED DATA")
        print(f"{'='*60}")
        
        programme_count = 0
        for programme, courses in extracted_data.items():
            if programme_count >= max_programmes:
                break
            
            print(f"\n🎓 {programme}:")
            
            if not courses:
                print("  No courses found")
                continue
            
            # Show first 5 courses
            course_count = 0
            for course_code, periods in courses.items():
                if course_count >= 5:
                    break
                print(f"  {course_code}: {periods}")
                course_count += 1
            
            if len(courses) > 5:
                print(f"  ... and {len(courses) - 5} more courses")
            
            programme_count += 1
        
        if len(extracted_data) > max_programmes:
            print(f"\n... and {len(extracted_data) - max_programmes} more programmes")
    
    def create_flattened_output(self, extracted_data: Dict[str, Dict[str, List[str]]]) -> Dict[str, Any]:
        """
        Create a flattened version of the extracted data with additional metadata.
        
        Args:
            extracted_data: Data extracted from programmes
            
        Returns:
            Dictionary with programmes, unique courses, and summary statistics
        """
        analysis = self.analyze_extracted_data(extracted_data)
        
        flattened_data = {
            "programmes": extracted_data,
            "all_unique_courses": analysis["all_unique_courses"],
            "summary": {
                "total_programmes": analysis["total_programmes"],
                "total_unique_courses": analysis["total_unique_courses"],
                "study_period_distribution": analysis["study_period_distribution"],
                "extraction_timestamp": self._get_timestamp()
            }
        }
        
        return flattened_data
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_predefined_programme_list(self) -> List[str]:
        """
        Get the predefined list of programme codes to extract.
        
        Returns:
            List of programme codes
        """
        return [
            # Master's Programmes (MP)
            "MPAEM",  # Materials Engineering, MSc Programme
            "MPALG",  # Computer Science - Algorithms, Languages and Logic, MSc Programme
            "MPAME",  # Applied Mechanics, MSc Programme
            "MPARC",  # Architecture and Urban Design, MSc Programme
            "MPBDP",  # Entrepreneurship and Business Design, MSc Programme
            "MPBIO",  # Biotechnology, MSc Programme
            "MPCAS",  # Complex Adaptive Systems, MSc Programme
            "MPCSN",  # Computer Systems and Networks, MSc Programme
            "MPDCM",  # Design and Construction Project Management, MSc Programme
            "MPDES",  # Industrial Design Engineering, MSc Programme
            "MPDSC",  # Data Science and Ai, MSc Programme
            "MPDSD",  # Architecture and Planning Beyond Sustainability, MSc Programme
            "MPEES",  # Embedded Electronic System Design, MSc Programme
            "MPENM",  # Engineering Mathematics and Computational Science, MSc Programme
            "MPEPO",  # Sustainable Electric Power Engineering and Electromobility, MSc Programme
            "MPHPC",  # High-Performance Computer Systems, MSc Programme
            "MPICT",  # Information and Communication Technology, MSc Programme
            "MPIDE",  # Interaction Design and Technologies, MSc Programme
            "MPIEE",  # Infrastructure and Environmental Engineering, MSc Programme
            "MPISC",  # Innovative and Sustainable Chemical Engineering, MSc Programme
            "MPLOL",  # Learning and Leadership, MSc Programme
            "MPMAR",  # Maritime Management, MSc Programme
            "MPMCN",  # Materials Chemistry, MSc Programme
            "MPMED",  # Biomedical Engineering, MSc Programme
            "MPMEI",  # Management and Economics Of Innovation, MSc Programme
            "MPMOB",  # Mobility Engineering, MSc Programme
            "MPNAT",  # Nanotechnology, MSc Programme
            "MPPDE",  # Product Development, MSc Programme
            "MPPEN",  # Production Engineering, MSc Programme
            "MPPHS",  # Physics, MSc Programme
            "MPQOM",  # Quality and Operations Management, MSc Programme
            "MPSCM",  # Supply Chain Management, MSc Programme
            "MPSEB",  # Structural Engineering and Building Technology, MSc Programme
            "MPSES",  # Sustainable Energy Systems, MSc Programme
            "MPSOF",  # Software Engineering and Technology, MSc Programme
            "MPSOV",  # Sound and Vibration, MSc Programme
            "MPSYS",  # Systems, Control and Mechatronics, MSc Programme
            "MPTSE",  # Industrial Ecology, MSc Programme
            "MPWPS",  # Wireless, Photonics and Space Engineering, MSc Programme
            
            # Bachelor's Programmes (TK)
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
    Main function to demonstrate usage of the ProgrammeCourseExtractor.
    
    This function shows how to:
    1. Initialize the extractor
    2. Extract course data for multiple programmes
    3. Generate analysis reports
    4. Save structured outputs
    """
    try:
        # Configuration
        TARGET_YEAR = "2024/2025"
        
        # Initialize the extractor
        extractor = ProgrammeCourseExtractor()
        
        # Get programme codes to extract
        programme_codes = extractor.get_predefined_programme_list()
        
        print(f"Programme Course Extractor")
        print(f"Target Year: {TARGET_YEAR}")
        print(f"Programmes to extract: {len(programme_codes)}")
        
        # Extract data for all programmes
        extracted_data = extractor.extract_multiple_programmes(programme_codes, TARGET_YEAR)
        
        # Generate analysis
        analysis = extractor.analyze_extracted_data(extracted_data)
        extractor.print_analysis_report(analysis)
        
        # Preview extracted data
        extractor.preview_extracted_data(extracted_data)
        
        # Save outputs
        print(f"\n{'='*60}")
        print("SAVING OUTPUTS")
        print(f"{'='*60}")
        
        # Save simple version
        output_file = os.path.join("Files_created", "course_codes_and_periods.json")
        extractor.save_to_json(extracted_data, output_file)
        
        # Save comprehensive version with analysis
        flattened_data = extractor.create_flattened_output(extracted_data)
        comprehensive_file = os.path.join("Files_created", "courses_in_program.json")
        extractor.save_to_json(flattened_data, comprehensive_file)
        
        print(f"\n{'='*60}")
        print("EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"📄 Simple output: {output_file}")
        print(f"📄 Comprehensive output: {comprehensive_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())