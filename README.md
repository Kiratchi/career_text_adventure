# 🎓 Chalmers Life Journey

An AI-powered interactive text adventure that simulates a realistic university journey at Chalmers University of Technology. Make meaningful choices that shape your academic path through all 15 engineering programs, from freshman year to graduation.

## ✨ Features

### 🤖 **AI-Powered Storytelling**
- **Dynamic Scenario Generation**: GPT-4 creates personalized scenarios based on your choices and program
- **Intelligent Director System**: Prevents repetitive scenarios and ensures variety
- **Contextual Decision Making**: Uses real Chalmers data to create authentic university experiences
- **Text-Based Prompt System**: Easy-to-edit prompts stored as simple text files

### 🎯 **Authentic University Experience**
- **15 Real Engineering Programs**: All actual Chalmers programs with accurate descriptions
- **5-Year Journey**: Progress through bachelor's and master's degrees
- **Real Data Integration**: Courses, companies, organizations, and facilities from Chalmers
- **Master's Program Selection**: Choose specialization in Year 3-4 based on your journey

### 🔄 **Smart Progression System**
- **Choice-Based Advancement**: 2 choices per academic year
- **Dynamic Year Progression**: Automatically advance based on experience
- **History Tracking**: AI remembers all your choices to avoid repetition
- **Personalized Recommendations**: Master's programs suggested based on your interests

### 🌐 **Modern Web Interface**
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Real-time Updates**: Immediate feedback on choices and progression
- **Program Search**: Filter through all 15 programs by keywords or fields
- **Visual Progress Tracking**: See your journey unfold with choice history

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- LiteLLM API access (supports multiple AI providers)
- Node.js (optional, for development)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/chalmers-life-journey.git
   cd chalmers-life-journey
   ```

2. **Install Python dependencies**
   ```bash
   pip install flask flask-cors python-dotenv litellm
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file
   echo "LITELLM_API_KEY=your_api_key_here" > .env
   echo "LITELLM_BASE_URL=https://your-litellm-endpoint.com" >> .env
   ```

4. **Create knowledge and prompts directories**
   ```bash
   mkdir knowledge
   mkdir prompts
   # Add your JSON files (see Knowledge Base section)
   # Add your prompt files (see Prompt System section)
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open your browser**
   ```
   http://localhost:5000
   ```

## 📁 Project Structure

```
chalmers-life-journey/
├── app.py                      # Main Flask backend
├── index.html                  # Frontend interface
├── .env                        # Environment variables
├── prompt_config.json          # AI prompt configuration
├── knowledge/                  # JSON knowledge base
│   ├── programs.json          # 15 Chalmers programs (REQUIRED)
│   ├── masters_programs.json  # Master's degree options
│   ├── chalmers_courses.json  # Course offerings by program
│   ├── gothenburg_companies.json # Local companies for internships
│   ├── student_organizations.json # Clubs and societies
│   ├── academic_calendar.json # University calendar
│   ├── campus_facilities.json # Libraries, labs, facilities
│   ├── swedish_university_info.json # University system info
│   ├── study_abroad_programs.json # Exchange opportunities
│   └── current_events.json    # Industry trends by field
├── prompts/                    # AI prompt text files
│   ├── director_analysis.txt  # Main scenario analysis
│   ├── scenario_creation.txt  # Scenario generation
│   ├── update_history_summary.txt # Profile updates
│   ├── update_personality_summary.txt # Character development
│   ├── update_current_situation.txt # Situation updates
│   ├── masters_selection_scenario.txt # Master's selection
│   ├── simple_scenario.txt    # Fallback scenarios
│   └── emergency_scenario.txt # Emergency fallbacks
└── README.md
```

## 🧠 Knowledge Base

The system uses JSON files to create realistic scenarios. Here's what each file contains:

### Core Files

#### `programs.json` (Required)
```json
{
  "chalmers_engineering_programs": [
    {
      "name": "Computer Science and Engineering",
      "description": "Develop tools that meet important societal challenges...",
      "field": "Computer Science",
      "keywords": ["computer science", "IT"]
    }
    // ... 14 more programs
  ]
}
```

#### `masters_programs.json`
```json
{
  "masters_programs": [
    {
      "name": "Robotics and Automation",
      "code": "ROBO",
      "description": "Advanced robotics systems...",
      "specializations": ["Industrial Robotics", "Autonomous Systems"],
      "career_paths": ["Robotics Engineer", "R&D Engineer"],
      "eligible_programs": ["Automation and Mechatronics", "Computer Science and Engineering"],
      "difficulty": "high"
    }
  ]
}
```

#### `chalmers_courses.json`
```json
{
  "Computer Science and Engineering": {
    "year_1": {
      "core": ["Programming Fundamentals", "Mathematics I"],
      "electives": ["Web Development", "Game Programming"]
    },
    "year_2": {
      "core": ["Data Structures", "Algorithms"],
      "electives": ["Machine Learning", "Database Systems"]
    }
  }
}
```

### Supporting Files

- **`gothenburg_companies.json`**: Local companies for internship scenarios
- **`student_organizations.json`**: Clubs and societies by category
- **`academic_calendar.json`**: Semester structure and important dates
- **`campus_facilities.json`**: Libraries, labs, and study spaces
- **`swedish_university_info.json`**: Grading system and university culture
- **`study_abroad_programs.json`**: Exchange and international opportunities
- **`current_events.json`**: Industry trends relevant to each program

## 📝 Text-Based Prompt System

The AI prompts are stored as simple text files for easy editing and maintenance.

### Prompt Configuration

#### `prompt_config.json`
```json
{
  "director_analysis": {
    "max_tokens": 1000,
    "temperature": 0.7,
    "description": "Analyzes student situation and decides what scenario to create next",
    "variables": ["student_summary", "program", "activities_done", "repetition_warnings"]
  },
  "scenario_creation": {
    "max_tokens": 1000,
    "temperature": 0.8,
    "description": "Creates the final scenario using AI with context from tools",
    "variables": ["student_name", "program", "technical_areas", "analysis"]
  }
}
```

### Prompt Text Files

#### `prompts/director_analysis.txt`
```
You are the Director of a university life simulation game. Your job is to decide what scenario should happen next for the student.

STUDENT PROFILE:
{student_summary}

PROGRAM-SPECIFIC CONTEXT:
Program: {program}
Activities completed: {activities_done}

REPETITION WARNINGS:
{repetition_warnings}

Create a scenario that offers NEW experiences and opportunities.
```

#### `prompts/scenario_creation.txt`
```
Create a realistic university life scenario for {student_name}.

STUDENT PROFILE:
{student_summary}

PROGRAM OPPORTUNITIES:
- Technical areas: {technical_areas}
- Career paths: {career_paths}

RESPOND WITH THIS EXACT JSON FORMAT:
{
  "title": "Brief descriptive title",
  "situation": "2-3 sentences explaining the situation",
  "options": [...]
}
```

### Editing Prompts

1. **Edit text files directly** in the `prompts/` directory
2. **Update configuration** in `prompt_config.json` if needed
3. **Reload prompts** without restarting server:
   ```bash
   curl -X POST http://localhost:5000/api/prompts/reload
   ```

## 🎮 How to Play

### 1. **Start Your Journey**
- Enter your name
- Choose from 15 authentic Chalmers engineering programs
- Each program has unique opportunities and career paths

### 2. **Make Meaningful Choices**
- Face realistic university scenarios every semester
- Choices range from course selection to career preparation
- Each decision shapes your character and future opportunities

### 3. **Progress Through Years**
- **Year 1-2**: Foundation and exploration
- **Year 3**: Specialization begins, choose master's program
- **Year 4-5**: Advanced studies and career preparation

### 4. **Experience Variety**
- AI ensures no repetitive scenarios
- Scenarios adapt to your program and previous choices
- Build relationships, join organizations, apply for internships

## 🛠 Technical Architecture

### Backend (Flask)
- **AI Director System**: Orchestrates scenario generation using text-based prompts
- **Tool-Based Architecture**: 15+ specialized tools for different scenario types
- **Profile Management**: Tracks student progression and history
- **Knowledge Integration**: Seamlessly uses JSON data in scenarios
- **Simple Prompt Loading**: Easy-to-edit text files for AI prompts

### Frontend (Vanilla JS + Modern CSS)
- **Responsive Design**: Works on all devices
- **Real-time Updates**: Immediate feedback on choices
- **Visual Progress Tracking**: See your journey unfold
- **Program Search**: Advanced filtering capabilities

### AI Integration
- **LiteLLM**: Supports multiple AI providers (OpenAI, Anthropic, etc.)
- **Context-Aware**: Uses your full history for personalized content
- **Repetition Avoidance**: Smart algorithms prevent boring repeated scenarios
- **Data-Driven**: All scenarios use real Chalmers information
- **Configurable Parameters**: Easy to adjust temperature and token limits

## 🔧 AI Director System

The core innovation is the AI Director that creates personalized scenarios:

### Director Tools
```python
- get_courses(): Course selection scenarios
- get_organizations(): Social and extracurricular opportunities  
- get_companies(): Internship and career preparation
- get_masters_programs(): Advanced degree selection
- analyze_progress(): Personalized development suggestions
- get_history_context(): Prevents repetitive scenarios
- check_prerequisites(): Ensures realistic progression
- predict_future_paths(): Career guidance scenarios
```

### Scenario Generation Process
1. **Analyze Student History**: What has the student already done?
2. **Load Analysis Prompt**: Get director analysis prompt from text file
3. **AI Analysis**: Determine what type of scenario to create
4. **Execute Tools**: Gather real university data based on analysis
5. **Load Scenario Prompt**: Get scenario creation prompt from text file
6. **Generate Scenario**: AI creates personalized, authentic scenario with 3-4 options

### Prompt System Features
- **Easy Editing**: Edit prompts in any text editor
- **Hot Reload**: Update prompts without restarting server
- **Version Control**: Track prompt changes in Git
- **Configurable**: Adjust AI parameters per prompt
- **Debugging**: Test prompts individually

## 🎯 Educational Value

### For Students
- **University Preparation**: Understand what university life is really like
- **Program Exploration**: Learn about different engineering fields
- **Career Awareness**: Discover career paths and preparation strategies
- **Decision Making**: Practice important life choices in a safe environment

### For Educators
- **Guidance Tool**: Help students understand university progression
- **Program Promotion**: Showcase the breadth of engineering education
- **Interactive Learning**: Engage students with realistic scenarios
- **Data-Driven Insights**: Based on real university information
- **Customizable Content**: Easy to modify prompts for different contexts

## 🌟 Advanced Features

### Master's Program Selection
- Triggered automatically in Year 3-4
- AI recommends programs based on your journey
- Influences final year scenarios and career paths
- Realistic selection process with prerequisites

### Smart Repetition Avoidance
```python
# Example: Student has already joined robotics team
# AI Director will suggest different activities:
- Drama society instead of another technical club
- Study abroad instead of another local opportunity  
- Startup incubator instead of another structured organization
```

### Dynamic Difficulty Scaling
- Early years: Basic choices about study habits and social life
- Later years: Complex decisions about specialization and career
- Adapts to student's demonstrated interests and strengths

### Comprehensive Progress Tracking
- **Academic Development**: Courses, grades, research
- **Social Growth**: Organizations, leadership, networking
- **Professional Preparation**: Internships, career planning
- **Personal Development**: Character traits and preferences

## 🔍 API Endpoints

### Game Management
- `POST /api/start_game`: Initialize new game session
- `POST /api/set_name`: Set player name
- `POST /api/set_program`: Choose engineering program
- `GET /api/get_profile`: Get current player state

### Gameplay
- `POST /api/generate_choice`: Get next scenario from AI Director
- `POST /api/make_choice`: Submit player choice and update profile

### Prompt Management
- `GET /api/prompts/list`: List all available prompts
- `POST /api/prompts/reload`: Reload all prompts from files
- `GET /api/prompts/<name>`: Get specific prompt details
- `PUT /api/prompts/<name>`: Update prompt content

### Debug & Admin
- `POST /api/reload_knowledge`: Refresh JSON knowledge files
- `GET /api/debug/sessions`: View active game sessions
- `GET /api/debug/director_tools`: Inspect available AI tools
- `GET /api/debug/prompts`: Debug prompt system status

## 🚀 Deployment Options

### Local Development
```bash
python app.py
# Access at http://localhost:5000
```

### Production Deployment
- **Heroku**: Add `Procfile` and `requirements.txt`
- **Docker**: Containerize with Flask + knowledge files + prompts
- **AWS/Google Cloud**: Deploy as web service
- **Self-hosted**: Use gunicorn + nginx

### Environment Variables
```env
LITELLM_API_KEY=your_key_here
LITELLM_BASE_URL=https://your-endpoint.com
FLASK_ENV=production
PORT=5000
```

## 🤝 Contributing

### Adding New Programs
1. Add program to `knowledge/programs.json`
2. Add courses to `knowledge/chalmers_courses.json`
3. Update master's programs in `knowledge/masters_programs.json`
4. Test scenario generation

### Improving AI Prompts
1. Edit prompt files in `prompts/` directory
2. Update configuration in `prompt_config.json`
3. Test with `/api/prompts/reload`
4. Version control your prompt changes

### Adding New Scenario Types
1. Create new prompt files for the scenario type
2. Add configuration to `prompt_config.json`
3. Create new director tools if needed
4. Test with different student profiles

### Frontend Enhancements
1. Improve mobile responsiveness
2. Add animations and visual feedback
3. Create better program discovery experience
4. Enhance progress visualization

## 🐛 Troubleshooting

### Common Issues

**"AI director not available"**
- Check LITELLM_API_KEY in `.env` file
- Verify LITELLM_BASE_URL is correct
- Ensure API endpoint is accessible

**"Prompt content is empty"**
- Check if `prompts/` directory exists
- Verify text files are present and readable
- Use `/api/prompts/list` to check loaded prompts

**"Knowledge file not found"**
- Verify `knowledge/` directory exists
- Check JSON file formatting
- Review file permissions

**"Session not found"**
- Clear browser cache
- Restart Flask application
- Check browser console for errors

### Debug Mode
```python
# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)

# Check prompt system
GET /api/debug/prompts

# Test specific prompt
GET /api/prompts/director_analysis

# Check available tools
GET /api/debug/director_tools
```

## 📊 Performance Considerations

### API Rate Limits
- LiteLLM supports multiple providers
- Implement caching for repeated scenarios
- Consider response streaming for better UX

### Memory Usage
- JSON files cached in memory
- Prompt files loaded on demand
- Session data grows with game length

### Scalability
- Stateless design enables horizontal scaling
- JSON knowledge base easily versioned
- Text prompts easily distributed and cached

## 🎓 Educational Applications

### Classroom Use
- **University Preparation Courses**: Help students understand university life
- **Engineering Career Exploration**: Discover different engineering fields  
- **Decision Making Skills**: Practice important life choices
- **Swedish University System**: Learn about European higher education
- **Prompt Engineering**: Teach students about AI interaction

### Self-Directed Learning
- **Program Research**: Explore Chalmers programs interactively
- **Career Planning**: Understand paths from education to career
- **University Culture**: Experience Swedish academic environment
- **Personal Development**: Reflect on interests and goals

## 🔧 Customization

### Adapting for Other Universities
1. **Update knowledge files** with your university's data
2. **Modify prompts** to reflect your university's culture
3. **Adjust program structure** in code if needed
4. **Test with university-specific scenarios**

### Changing AI Providers
1. **Update environment variables** for new provider
2. **Modify LiteLLM configuration** if needed
3. **Test prompt compatibility** with new model
4. **Adjust temperature/token settings** in `prompt_config.json`

### Adding New Languages
1. **Create translated prompt files** in `prompts/[language]/`
2. **Add language detection** to prompt loading
3. **Update knowledge files** with localized content
4. **Test scenarios** in new language