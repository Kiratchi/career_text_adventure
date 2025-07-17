# 🎓 Chalmers Life Journey

An interactive AI-powered text adventure game that simulates a complete university journey through Chalmers University of Technology. Players make meaningful choices that shape their academic path, personality, and future opportunities using real university data and AI-generated scenarios.

## 🌟 Project Overview

Chalmers Life Journey is a sophisticated educational simulation that combines:
- **Real University Data** - Authentic information about Chalmers programs, courses, and student life
- **AI-Powered Storytelling** - Dynamic scenario generation using LLM technology
- **Meaningful Choices** - Every decision impacts the player's journey and future opportunities
- **Progressive Narrative** - From first-year orientation to thesis defense and career preparation
- **Personalized Experience** - AI adapts scenarios based on player choices and personality development

## 🎮 Game Features

### Core Gameplay
- **10 Major Life Choices** spanning 5 years of university life
- **16 Engineering Programs** with authentic Chalmers course data
- **Master's Program Selection** with real specialization options
- **Dynamic Personality Development** tracked by AI throughout the journey
- **Multiple Scenario Types**: Mottagning (orientation), exchange opportunities, thesis projects, career preparation

### Technical Features
- **AI Director System** - Intelligent scenario generation and choice consequences
- **Real-time Profile Updates** - Comprehensive history and personality tracking
- **Debug Interface** - Complete visibility into AI decision-making process
- **Responsive Design** - Beautiful UI that works on desktop and mobile
- **End-to-End Journey** - Complete experience from enrollment to graduation

## 🗂️ Project Structure

```
career text adventure v2/
├── 📁 knowledge/                    # University data and information
│   ├── 📁 student_sections/         # Student section information by program
│   │   ├── A.txt                    # Architecture students
│   │   ├── D.txt                    # Computer Science students
│   │   ├── E.txt                    # Electrical Engineering students
│   │   └── [14 other sections...]
│   ├── courses_in_program.json      # Course-program mappings
│   ├── course_summary_simplified.json # Course descriptions and summaries
│   ├── masters_programs.json        # Master's program information
│   ├── programs.json                # Bachelor program details
│   ├── program_master_bidirectional_mapping.json # Program progression paths
│   ├── sports.txt                   # Sports and recreation information
│   ├── studies.txt                  # Study environment and campus info
│   └── exchange_year.txt            # International exchange opportunities
├── 📁 prompts/                      # AI prompt templates
│   ├── director_analysis.txt        # AI scenario decision-making
│   ├── scenario_creation.txt        # Dynamic scenario generation
│   ├── thesis_analysis.txt          # Thesis project scenarios
│   ├── career_analysis.txt          # Career preparation scenarios
│   ├── exchange_analysis.txt        # Exchange opportunity scenarios
│   ├── mottagning_analysis.txt      # Orientation scenarios
│   ├── introduction.txt             # Personalized introductions
│   ├── masters_introduction_analysis.txt # Master's program guidance
│   └── [profile update prompts...]
├── 📁 Knowledge_data_creation/      # Data processing pipeline
│   ├── 📁 Files/                    # Raw Chalmers data
│   ├── 📁 Files_created/            # Processed knowledge files
│   ├── course_data_processor.py     # Course data processing with AI
│   ├── program_analyzer.py          # Program analysis and categorization
│   ├── get_masters_info.py          # Master's program data extraction
│   └── program_course_extractor.py  # Course-program relationship mapping
├── app.py                           # Flask backend with AI Director
├── index.html                       # Frontend game interface
├── debug.html                       # AI debugging interface
├── .env                             # Environment configuration
└── requirements.txt                 # Python dependencies
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- Access to LLM API (configured for LiteLLM)
- Modern web browser

### Quick Start
1. **Clone and Install**
```bash
git clone <repository-url>
cd "career text adventure v2"
pip install -r requirements.txt
```

2. **Configure Environment**
Create `.env` file with your LLM API credentials:
```env
LITELLM_API_KEY=your_api_key_here
LITELLM_BASE_URL=https://anast.ita.chalmers.se:4000
```

3. **Launch the Game**
```bash
python app.py
```

4. **Start Playing**
Open your browser to `http://localhost:5000` and begin your Chalmers journey!

## 🎯 Game Flow

### Year 1: Foundation
- **Choice 0**: Mottagning (Orientation) - First university experience
- **Choice 1**: Study approach and social integration

### Year 2-3: Development  
- **Choices 2-4**: Academic focus, extracurriculars, skill development
- **Choice 5**: 🎓 **Master's Program Selection** - Choose your specialization

### Year 4-5: Specialization
- **Choices 6-8**: Advanced studies, possible exchange programs
- **Choice 9**: 📝 **Thesis Project Selection** - Define your research
- **Choice 10**: 💼 **Career Preparation** - Final choice before graduation

### Graduation
- **End Screen**: Comprehensive journey summary with option to replay

## 🛠️ Technical Architecture

### Backend (Flask + AI)
- **AI Director**: Intelligent scenario generation based on player choices
- **Profile Management**: Dynamic personality and history tracking
- **Knowledge System**: Real university data integration
- **Tool System**: Modular information sources (sports, studies, courses, exchange)
- **Debug Logging**: Complete AI decision-making transparency

### Frontend (Vanilla JS + CSS)
- **Interactive Story Interface**: Choice-driven narrative progression
- **Profile Tracking**: Real-time display of player journey
- **Master's Program Modal**: Enhanced selection interface
- **Responsive Design**: Beautiful UI across all devices
- **End Game Celebration**: Satisfying journey completion

### AI Integration
- **LiteLLM**: Multi-model LLM support (Claude, GPT-4)
- **Dynamic Prompting**: Context-aware scenario generation
- **Personality Tracking**: AI-powered character development
- **Comprehensive History**: Detailed journey narration

## 📊 Game Statistics

Based on real Chalmers data (2024/2025):
- **🎓 Programs**: 16 authentic engineering programs
- **📚 Courses**: 1,250+ real course descriptions
- **🎯 Master's Options**: 35+ specialized master's programs
- **📋 Student Sections**: 14 authentic student communities
- **🌍 Exchange**: Real international opportunities
- **⚡ Scenarios**: Unlimited AI-generated unique experiences

## 🔧 Configuration

### AI Models
Supports multiple LLM providers through LiteLLM:
```python
models_to_try = [
    "claude-haiku-3.5", 
    "claude-sonnet-3.7", 
    "gpt-4.1-2025-04-14"
]
```

### Scenario Triggers
- **Mottagning**: Choice 0 (Year 1)
- **Master's Selection**: Choice 5 (Year 3)
- **Exchange Opportunity**: Choice 6/8 (50% chance, Year 4)
- **Thesis Selection**: Choice 9 (Year 5)
- **Career Preparation**: Choice 10 (Year 5)

### Debug Interface
Access detailed AI insights at `http://localhost:5000/debug`:
- Scenario generation process
- LLM call analysis  
- Tool execution tracking
- Profile state management
- Quick game state setup

## 📈 Data Processing Pipeline

The project includes a complete data processing system:

### Knowledge Creation Tools
- **Course Processor**: AI-generated course summaries
- **Program Analyzer**: Degree categorization and mapping
- **Master's Extractor**: Graduate program relationships
- **Section Mapper**: Student community information

### Data Sources
- Official Chalmers course catalogs
- Program syllabi and requirements
- Student section descriptions
- Exchange program information
- Campus facilities and activities

## 🎮 Gameplay Examples

### Choice Impact System
```
Player chooses: "Focus on getting the best grades possible"
→ Personality: becomes more grade-focused
→ Future scenarios: Academic excellence opportunities
→ Master's options: Research-oriented programs prioritized
```

### Dynamic Storytelling
```
"As a Computer Science student in Year 3, your excellent 
grades in algorithms courses have caught the attention of 
Professor Anderson. She's offered you three paths..."
```

### Master's Selection
```
Personalized AI guidance → Modal interface → Real programs
→ Career impact analysis → Informed decision making
```

## 🔍 Debug Features

- **Real-time AI Analysis**: See how the AI Director makes decisions
- **Profile Inspection**: Complete player state visibility  
- **Quick Setup Tools**: Jump to any game state for testing
- **Scenario Triggers**: Understand when special events occur
- **Performance Metrics**: LLM call timing and success rates

## 🌟 Educational Value

This project demonstrates:
- **AI Integration**: Practical LLM application in interactive media
- **Data Processing**: Real-world data pipeline construction
- **Web Development**: Modern full-stack application architecture
- **Game Design**: Choice-driven narrative systems
- **Educational Technology**: Immersive learning experiences

## 📝 Contributing

The project welcomes contributions in:
- Additional scenario types and prompts
- Enhanced AI Director logic
- New knowledge sources and tools
- UI/UX improvements
- Performance optimizations

## 🏆 Achievements

This project successfully creates:
- ✅ **Engaging Educational Experience** - Real university simulation
- ✅ **Advanced AI Integration** - Sophisticated LLM-powered gameplay  
- ✅ **Comprehensive Data Usage** - Authentic university information
- ✅ **Professional Architecture** - Scalable, maintainable codebase
- ✅ **Complete User Journey** - From enrollment to graduation
- ✅ **Debugging Excellence** - Full visibility into AI decision-making

---

*Embark on your personalized journey through Chalmers University of Technology. Every choice matters, every path is unique, and your story is entirely your own.*

**🎯 Ready to begin? Launch the application and discover where your choices will take you!**