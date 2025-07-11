# University Data Analysis Pipeline

A comprehensive Python-based data analysis pipeline for processing and analyzing university programme and course data from Chalmers University of Technology.

## 📋 Project Overview

This project provides tools to analyze and process university data including:
- **Course Information** - Detailed course data with AI-generated summaries
- **Programme Structure** - Programme categorization and accreditation mapping
- **Course-Programme Mapping** - Relationships between courses and programmes
- **Study Period Analysis** - Course scheduling and timing information

## 🗂️ Project Structure

```
OneDrive_1_10-07-2025/
├── Files/                          # Input data files
│   ├── pew_courses_en-*.json       # Course data
│   ├── pew_programmesyllabuses_en-*.json  # Programme syllabi
│   ├── pew_studyprogrammes_en-*.json      # Study programme data
│   └── [other data files...]
├── Files_created/                  # Generated output files
│   ├── accredited_masters.json     # Programme accreditation mapping
│   ├── courses_in_program.json     # Course-programme relationships
│   ├── course_summary_full.json    # Complete course data with AI summaries
│   └── course_summary_simplified.json  # Simplified course information
├── course_data_processor.py        # Main course data processing
├── program_analyzer.py             # Programme analysis and categorization
├── program_course_extractor.py     # Course-programme relationship extraction
├── .env                            # Environment variables
└── .gitignore                      # Git ignore file
```

## 🚀 Features

### 1. Course Data Processing (`course_data_processor.py`)
- **AI-Powered Summaries**: Generates concise course descriptions using LLM
- **Batch Processing**: Handles large datasets with progress tracking
- **Flexible Output**: Creates both detailed and simplified course datasets
- **Error Handling**: Robust processing with comprehensive error reporting

### 2. Programme Analysis (`program_analyzer.py`)
- **Degree Categorization**: Automatically categorizes programmes by degree type
- **Accreditation Mapping**: Extracts master's programme accreditations for bachelor programmes
- **Search Functionality**: Find programmes by name or code
- **Comprehensive Reporting**: Detailed statistics and analysis

### 3. Programme-Course Extraction (`program_course_extractor.py`)
- **Course Mapping**: Maps courses to their respective programmes
- **Study Period Analysis**: Identifies when courses are offered
- **Batch Processing**: Handles multiple programmes simultaneously
- **Statistical Analysis**: Provides insights into course distribution

## 📊 Output Files

### `accredited_masters.json`
Maps bachelor programmes to their accredited master programmes:
```json
{
  "programme_accreditations": {
    "TKDAT": ["MPALG", "MPCAS", "MPCSN", "MPDSC", ...],
    "TKELT": ["MPCAS", "MPCSN", "MPDSC", "MPEES", ...]
  },
  "all_accredited_masters": ["MPALG", "MPCAS", ...],
  "summary": {
    "total_programmes": 17,
    "programmes_with_accreditations": 12,
    "total_unique_accredited_masters": 35,
    "academic_year": "2024/2025"
  }
}
```

### `courses_in_program.json`
Programme-course relationships with study periods:
```json
{
  "programmes": {
    "MPBIO": {
      "BBT012": ["LP1"],
      "BBT051": ["LP1"],
      "KBB032": ["LP1"]
    }
  },
  "all_unique_courses": ["BBT001", "BBT012", ...],
  "summary": {
    "total_programmes": 56,
    "total_unique_courses": 1250
  }
}
```

### `course_summary_simplified.json`
Clean course data with AI summaries:
```json
[
  {
    "courseCode": "FSP047",
    "name": "English for engineers",
    "AI_summary": "This first-cycle course is designed for students...",
    "study_periods": ["LP1", "LP2"]
  }
]
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Required packages: `python-dotenv`, `litellm`, `tqdm`

### Environment Setup
1. Install dependencies:
```bash
pip install python-dotenv litellm tqdm
```

2. Configure environment variables in `.env`:
```env
LITELLM_API_KEY=your_api_key_here
LITELLM_BASE_URL=https://anast.ita.chalmers.se:4000
```

3. Ensure data files are in the `Files/` directory

## 🚦 Usage

### Course Data Processing
```bash
python course_data_processor.py
```
- Processes course data and generates AI summaries
- Creates both detailed and simplified outputs
- Configurable for dry-run testing

### Programme Analysis
```bash
python program_analyzer.py
```
- Analyzes programme structure and categorization
- Extracts accreditation relationships
- Generates comprehensive reports

### Programme-Course Extraction
```bash
python program_course_extractor.py
```
- Maps courses to programmes
- Analyzes study period distribution
- Creates relationship mappings

## 📈 Key Statistics

Based on 2024/2025 academic year data:
- **🎓 Programmes**: 72 total programmes across 9 degree categories
- **📚 Courses**: 1,250+ unique courses
- **🔗 Relationships**: 56 programmes mapped to course structures
- **🎯 Accreditations**: Bachelor programmes mapped to 35+ master programmes

## 🔧 Configuration

### Target Academic Year
Change the target year in each script:
```python
TARGET_YEAR = "2024/2025"  # Modify as needed
```

### Programme Selection
Modify programme lists in the extractors:
```python
programmes_to_extract = [
    "MPBIO",  # Biotechnology
    "TKDAT",  # Computer Science
    # Add more programmes...
]
```

### AI Model Configuration
Adjust LLM settings in `course_data_processor.py`:
```python
DEFAULT_MODEL = "gpt-4.1-2025-04-14"
```

## 📝 Data Sources

The project processes official Chalmers University data including:
- Course information and descriptions
- Programme syllabi and structure
- Study programme definitions
- Accreditation relationships


---

*This project demonstrates advanced data processing techniques for educational data analysis, featuring AI integration, comprehensive error handling, and scalable architecture.*