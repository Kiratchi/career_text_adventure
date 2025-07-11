"""
Course Data Processor

This script processes university course data to:
1. Generate AI summaries for courses using LLM
2. Extract simplified course information with study periods
3. Create clean JSON outputs for further analysis

Requirements:
- Python packages: python-dotenv, litellm, tqdm
- Environment variables: LITELLM_API_KEY, LITELLM_BASE_URL
- Input files: course data JSON in 'Files' directory

Usage:
    python course_processor.py
"""

import os
from dotenv import load_dotenv
import litellm
import json
import pprint
import time
from tqdm import tqdm  
from typing import List, Dict, Any, Optional


class CourseDataProcessor:
    """
    A class to process course data, generate AI summaries, and create simplified outputs.
    """
    
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        """
        Initialize the course processor.
        
        Args:
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
        
        # Load course data
        self.course_data = self._load_course_data()
        
        if not self.api_key:
            print("⚠️  Warning: No API key found. AI summarization will not work.")
    
    def _load_course_data(self) -> Dict[str, Any]:
        """
        Load course data from JSON file.
        
        Returns:
            Dictionary containing course data
            
        Raises:
            FileNotFoundError: If course data file doesn't exist
        """
        json_path = os.path.join('Files', 'pew_courses_en-20250623_061647.json')
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"Course data file not found: {json_path}\n"
                "Please ensure the course data file is in the 'Files' directory."
            )
        
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_course_info(self, course_code: str, ac_year: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get course information for a specific course code and academic year.
        
        Args:
            course_code: The course code to search for
            ac_year: Academic year (e.g., "2024/2025"). If None, returns most recent.
            
        Returns:
            Dictionary containing course information, or None if not found
        """
        matching = [
            doc['_source'] for doc in self.course_data['hits']['hits']
            if doc['_source'].get('courseCode', '').lower() == course_code.lower()
        ]

        if not matching:
            return None

        if ac_year:
            filtered = [c for c in matching if c.get('acYear') == ac_year]
            if not filtered:
                return None
            return filtered[0]
        else:
            # Sort by academic year descending to get most recent
            def ac_year_sort_key(course):
                try:
                    return int(course.get("acYear", "0/0").split("/")[0])
                except:
                    return 0
            return sorted(matching, key=ac_year_sort_key, reverse=True)[0]

    def summarize_course(self, course_description: str) -> str:
        """
        Generate an AI summary of a course using LLM.
        
        Args:
            course_description: Full course description to summarize
            
        Returns:
            AI-generated summary of the course
            
        Raises:
            ValueError: If course_description is empty
            RuntimeError: If LLM call fails
        """
        if not course_description:
            raise ValueError("Missing course_description")

        prompt = (
            "You are generating structured course descriptions to help an AI create stories in a university simulation. "
            "Summarize the following university course in one factual and concise paragraph. Include what the course covers, how it builds on prior knowledge, and what types of tasks or projects students engage in. "
            "Avoid mentioning the course name."
            f"\n\n{course_description}"
        )

        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                api_key=self.api_key,
                max_tokens=250,
                temperature=0.5,
                base_url=self.base_url
            )

            summary = response.choices[0].message.content.strip()
            return summary

        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}")

    def create_summary_copy_to_file(
        self,
        target_year: str,
        output_path: str,
        max_courses: Optional[int] = None,
        dry_run: bool = True
    ) -> None:
        """
        Create a copy of course data with AI summaries added.
        
        Args:
            target_year: Academic year to filter courses (e.g., "2024/2025")
            output_path: Path where to save the output JSON file
            max_courses: Maximum number of courses to process (None for all)
            dry_run: If True, adds placeholder summaries instead of calling LLM
        """
        start_time = time.time()
        filtered = []
        count = 0

        # Pre-filter matching courses to get accurate total
        matching_courses = [
            doc for doc in self.course_data['hits']['hits']
            if doc.get('_source', {}).get('acYear') == target_year
        ]
        total = len(matching_courses)
        
        mode = "dry run" if dry_run else "AI summarization"
        print(f"\nMatched {total} courses for year {target_year}. Starting {mode}...\n")

        # Iterate with tqdm for progress bar
        for doc in tqdm(matching_courses, total=total, desc="Processing courses", unit="course"):
            source = doc.get('_source', {})
            new_entry = source.copy()

            if dry_run:
                new_entry['AI_summary'] = "(summary placeholder)"
            else:
                try:
                    description = pprint.pformat(new_entry, width=120, compact=True)
                    new_entry['AI_summary'] = self.summarize_course(description)
                except Exception as e:
                    new_entry['AI_summary'] = f"Error summarizing: {e}"

            filtered.append(new_entry)
            count += 1

            if max_courses and count >= max_courses:
                break

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the processed data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered, f, indent=2, ensure_ascii=False)

        total_time = time.time() - start_time
        print(f"\n✔ Done. Saved {len(filtered)} courses to {output_path}")
        print(f"⏱ Total time: {total_time / 60:.2f} minutes")

    def extract_study_periods_with_credits(self, course_data: Dict[str, Any]) -> List[str]:
        """
        Extract study periods where credits are allocated from a course.
        
        Args:
            course_data: Dictionary containing course information
            
        Returns:
            List of study periods where credits are allocated (e.g., ['LP1', 'LP2'])
        """
        study_periods = []
        
        # Look through all course rounds
        for course_round in course_data.get('courseRounds', []):
            # Look through all moments in each round
            for moment in course_round.get('moments', []):
                # Check the distribution for periods with credits
                for dist in moment.get('distribution', []):
                    period_code = dist.get('code', '')
                    credit = dist.get('credit', 0)
                    
                    # Only include periods that have credits allocated
                    if credit and credit > 0 and period_code not in study_periods:
                        study_periods.append(period_code)
        
        return sorted(study_periods)

    def create_course_summary_json(
        self,
        input_file: str,
        output_file: str
    ) -> None:
        """
        Create a simplified JSON with courseCode, name, AI_summary, and study_periods.
        
        Args:
            input_file: Path to the full course data JSON with AI summaries
            output_file: Path where to save the simplified JSON
        """
        # Check if input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(
                f"Input file not found: {input_file}\n"
                "Please run the AI summarization step first."
            )
        
        # Load the full course data
        with open(input_file, 'r', encoding='utf-8') as f:
            courses_data = json.load(f)
        
        simplified_courses = []
        
        for course in courses_data:
            # Extract study periods with credits
            study_periods = self.extract_study_periods_with_credits(course)
            
            # Create simplified course entry
            simplified_course = {
                "courseCode": course.get('courseCode', ''),
                "name": course.get('name', ''),
                "AI_summary": course.get('AI_summary', ''),
                "study_periods": study_periods
            }
            
            simplified_courses.append(simplified_course)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save the simplified data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(simplified_courses, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Successfully created simplified course summary with {len(simplified_courses)} courses")
        print(f"✓ Saved to: {output_file}")
        
        # Print some statistics
        self._print_study_period_stats(simplified_courses)

    def _print_study_period_stats(self, courses_data: List[Dict[str, Any]]) -> None:
        """Print statistics about study period distribution."""
        period_counts = {}
        for course in courses_data:
            for period in course['study_periods']:
                period_counts[period] = period_counts.get(period, 0) + 1
        
        print(f"\nStudy period distribution:")
        for period, count in sorted(period_counts.items()):
            print(f"  {period}: {count} courses")

    def preview_courses(self, courses_data: List[Dict[str, Any]], num_courses: int = 3) -> None:
        """
        Preview the first few courses to verify the extraction.
        
        Args:
            courses_data: List of course dictionaries
            num_courses: Number of courses to preview
        """
        print(f"\nPreview of first {num_courses} courses:")
        print("-" * 50)
        
        for i, course in enumerate(courses_data[:num_courses]):
            print(f"\nCourse {i+1}:")
            print(f"  Code: {course['courseCode']}")
            print(f"  Name: {course['name']}")
            print(f"  Study Periods: {course['study_periods']}")
            print(f"  AI Summary: {course['AI_summary'][:100]}...")

    def print_dataset_stats(self) -> None:
        """Print basic statistics about the loaded dataset."""
        total_hits = self.course_data['hits']['total']['value']
        returned_docs = len(self.course_data['hits']['hits'])
        unique_codes = {
            doc['_source'].get('courseCode') 
            for doc in self.course_data['hits']['hits'] 
            if '_source' in doc
        }
        
        print(f"Dataset Statistics:")
        print(f"  Total hits: {total_hits}")
        print(f"  Returned documents: {returned_docs}")
        print(f"  Unique course codes: {len(unique_codes)}")


def main():
    """
    Main function to demonstrate usage of the CourseDataProcessor.
    
    This function shows how to:
    1. Initialize the processor
    2. Generate AI summaries for courses
    3. Create simplified course summaries
    4. Preview the results
    """
    try:
        # Initialize the processor
        processor = CourseDataProcessor()
        
        # Print dataset information
        processor.print_dataset_stats()
        
        # Configuration
        TARGET_YEAR = "2024/2025"
        DRY_RUN = False  # Set to True to test without calling LLM
        
        # File paths
        full_summary_file = os.path.join("Files_created", "course_summary_full.json")
        simplified_summary_file = os.path.join("Files_created", "course_summary_simplified.json")
        
        # Step 1: Generate AI summaries (uncomment to run)
        print(f"\n{'='*60}")
        print("STEP 1: Generate AI Summaries")
        print(f"{'='*60}")
        
        # Uncomment the following lines to generate AI summaries
        # processor.create_summary_copy_to_file(
        #     target_year=TARGET_YEAR,
        #     output_path=full_summary_file,
        #     dry_run=DRY_RUN
        # )
        print("AI summary generation is commented out. Uncomment to run.")
        
        # Step 2: Create simplified course summaries
        print(f"\n{'='*60}")
        print("STEP 2: Create Simplified Course Summaries")
        print(f"{'='*60}")
        
        if os.path.exists(full_summary_file):
            processor.create_course_summary_json(full_summary_file, simplified_summary_file)
            
            # Step 3: Preview the results
            print(f"\n{'='*60}")
            print("STEP 3: Preview Results")
            print(f"{'='*60}")
            
            with open(simplified_summary_file, 'r', encoding='utf-8') as f:
                simplified_data = json.load(f)
            
            processor.preview_courses(simplified_data)
        else:
            print(f"❌ Full summary file not found: {full_summary_file}")
            print("Please run the AI summarization step first by uncommenting the relevant lines.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())