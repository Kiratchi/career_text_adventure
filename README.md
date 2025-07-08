# 🎓 Chalmers Life Journey

An AI-powered interactive text adventure that simulates a realistic university journey at Chalmers University of Technology. Make meaningful choices that shape your academic path through all 15 engineering programs, from freshman year to graduation.

## ✨ Features

### 🤖 **AI-Powered Storytelling**
- **Dynamic Scenario Generation**: GPT-4 creates personalized scenarios based on your choices and program
- **Intelligent Director System**: Prevents repetitive scenarios and ensures variety
- **Contextual Decision Making**: Uses real Chalmers data to create authentic university experiences

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
- OpenAI API key
- Node.js (optional, for development)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/chalmers-life-journey.git
   cd chalmers-life-journey
   ```

2. **Install Python dependencies**
   ```bash
   pip install flask flask-cors python-dotenv langchain-openai
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

4. **Create knowledge directory and add JSON files**
   ```bash
   mkdir knowledge
   # Add your JSON files (see Knowledge Base section)
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
- **AI Director System**: Orchestrates scenario generation
- **Tool-Based Architecture**: 15+ specialized tools for different scenario types
- **Profile Management**: Tracks student progression and history
- **Knowledge Integration**: Seamlessly uses JSON data in scenarios

### Frontend (Vanilla JS + Modern CSS)
- **Responsive Design**: Works on all devices
- **Real-time Updates**: Immediate feedback on choices
- **Visual Progress Tracking**: See your journey unfold
- **Program Search**: Advanced filtering capabilities

### AI Integration
- **GPT-4 Mini**: Powers scenario generation
- **Context-Aware**: Uses your full history for personalized content
- **Repetition Avoidance**: Smart algorithms prevent boring repeated scenarios
- **Data-Driven**: All scenarios use real Chalmers information

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
2. **Identify Growth Areas**: What new experiences would be valuable?
3. **Gather Real Data**: Use appropriate tools to get university information
4. **Generate Scenario**: AI creates personalized, authentic scenario
5. **Present Choices**: 3-4 meaningful options that impact the journey

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

### Debug & Admin
- `POST /api/reload_knowledge`: Refresh JSON knowledge files
- `GET /api/debug/sessions`: View active game sessions
- `GET /api/debug/director_tools`: Inspect available AI tools

## 🚀 Deployment Options

### Local Development
```bash
python app.py
# Access at http://localhost:5000
```

### Production Deployment
- **Heroku**: Add `Procfile` and `requirements.txt`
- **Docker**: Containerize with Flask + knowledge files
- **AWS/Google Cloud**: Deploy as web service
- **Self-hosted**: Use gunicorn + nginx

### Environment Variables
```env
OPENAI_API_KEY=your_key_here
FLASK_ENV=production
PORT=5000
```

## 🤝 Contributing

### Adding New Programs
1. Add program to `knowledge/programs.json`
2. Add courses to `knowledge/chalmers_courses.json`
3. Update master's programs in `knowledge/masters_programs.json`
4. Test scenario generation

### Improving AI Scenarios
1. Enhance prompt engineering in `GameDirector`
2. Add new tools for specific scenario types
3. Improve context awareness and variety
4. Test with different student profiles

### Frontend Enhancements
1. Improve mobile responsiveness
2. Add animations and visual feedback
3. Create better program discovery experience
4. Enhance progress visualization

## 🐛 Troubleshooting

### Common Issues

**"AI director not available"**
- Check OpenAI API key in `.env` file
- Verify internet connection
- Ensure sufficient API credits

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

# Check available tools
GET /api/debug/director_tools

# Monitor active sessions  
GET /api/debug/sessions
```

## 📊 Performance Considerations

### API Rate Limits
- OpenAI API: ~60 requests/minute
- Implement caching for repeated scenarios
- Consider response streaming for better UX

### Memory Usage
- JSON files cached in memory
- Session data grows with game length
- Consider session cleanup for production

### Scalability
- Stateless design enables horizontal scaling
- JSON knowledge base easily versioned
- AI scenarios cached and reusable

## 🎓 Educational Applications

### Classroom Use
- **University Preparation Courses**: Help students understand university life
- **Engineering Career Exploration**: Discover different engineering fields  
- **Decision Making Skills**: Practice important life choices
- **Swedish University System**: Learn about European higher education

### Self-Directed Learning
- **Program Research**: Explore Chalmers programs interactively
- **Career Planning**: Understand paths from education to career
- **University Culture**: Experience Swedish academic environment
- **Personal Development**: Reflect on interests and goals

## 📝 License

MIT License - feel free to use, modify, and distribute.

## 🙏 Acknowledgments

- **Chalmers University of Technology**: For inspiring this educational simulation
- **OpenAI**: For powering the AI Director system
- **Flask Community**: For the excellent web framework
- **Contributors**: Everyone who helped make this project possible

---

**Ready to start your Chalmers journey? 🚀**

[Get Started](#quick-start) | [View Demo](http://localhost:5000) | [Report Issues](https://github.com/yourusername/chalmers-life-journey/issues)