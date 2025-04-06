# HR Assistant: Agentic AI App for Hiring Process Planning
This is a Flask-based application that uses LangChain Agents to assist HR professionals in planning and managing the hiring process for startups.

## Features

- Job Description Generator: Creates detailed job descriptions based on role requirements and company information
- Interview Process Planner: Designs a comprehensive interview process with stages, timelines, and team assignments
- Candidate Evaluation Criteria Generator: Develops customized evaluation criteria for any role
- Hiring Analytics Generator: Provides insights and projections for the hiring process
- Conversation Memory: Retains context throughout the interaction
- Multi-step Reasoning: Uses LangChain agents to break down complex HR planning tasks
- Interactive Frontend: User-friendly interface with real-time chat capabilities

### Project Structure

```
hr-app/
├── app.py                # Main Flask application
├── templates/
│   └── index.html        # Frontend HTML template
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

# Installation

Clone the repository
Create a virtual environment:
```python -m venv venv```
```source venv/bin/activate  # On Windows: venv\Scripts\activate```

Install dependencies:
```pip install -r requirements.txt```

- Usage

Run the Flask application:
```python app.py```

Open your web browser and navigate to ```http://127.0.0.1:5000```

#### Interact with the HR Assistant by typing in the chat interface
#### Use the suggested quick prompts to explore different functionalities

### How It Works
The application leverages LangChain Agents with the following components:

- Tools: Four specialized tools for different aspects of the hiring process
- Memory: Conversation buffer memory to maintain context
- Agent: A conversational agent that coordinates tool usage
- Frontend: Interactive UI for user engagement

Example Prompts

1. "Create a job description for a Senior Software Engineer role with 5+ years of experience."
2. "Plan an interview process for a Marketing Manager position with a timeline of 6 weeks."
3. "Develop evaluation criteria for a Product Manager focusing on user research skills."
4. "Generate hiring analytics for a UX Designer position with approximately 100 expected applicants."
