import os
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage


app = Flask(__name__)
app.secret_key = os.urandom(24)

openai_api_key = os.getenv("OPENAI_API_KEY", "your key here") 
llm = ChatOpenAI(
    temperature=0.7,
    model_name="gpt-4",
    openai_api_key=openai_api_key
)


memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)


system_message = SystemMessage(
    content=(
        "You are an AI assistant helping HR professionals plan and manage "
        "hiring processes for a startup. You have access to tools that can "
        "generate job descriptions, plan interviews, define evaluation criteria, "
        "and provide hiring analytics. Use multi-step reasoning and these tools "
        "to assist the user thoroughly."
    )
)
memory.chat_memory.add_message(system_message)

@tool
def generate_job_description(input_str: str) -> str:
    """
    Generate a detailed job description. Input is a JSON string with
    { "role": "", "requirements": "", "company_info": "" }
    """
    try:
        data = json.loads(input_str)
        role = data.get("role", "")
        requirements = data.get("requirements", "")
        company_info = data.get("company_info", "")

        prompt = PromptTemplate(
            input_variables=["role", "requirements", "company_info"],
            template="""
Create a compelling job description for the role of {role} at a startup.

Requirements: {requirements}
Company Information: {company_info}

Generate a detailed job description including:
1. Role overview
2. Responsibilities
3. Required qualifications
4. Preferred skills
5. Benefits and perks
"""
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(
            role=role,
            requirements=requirements,
            company_info=company_info
        )
        return result.strip()
    except Exception as e:
        return f"Error generating job description: {str(e)}"


@tool
def plan_interview_process(input_str: str) -> str:
    """
    Create an interview process plan. Input is a JSON string with
    { "role": "", "team_size": "", "timeline": "" }
    """
    try:
        data = json.loads(input_str)
        role = data.get("role", "")
        team_size = data.get("team_size", "")
        timeline = data.get("timeline", "")

        prompt = PromptTemplate(
            input_variables=["role", "team_size", "timeline"],
            template="""
Create an interview process plan for hiring a {role} at a startup.

Team Size: {team_size}
Timeline: {timeline}

Generate a detailed interview process including:
1. Screening stages
2. Interview rounds
3. Assessments or tasks
4. Timeline for each stage
5. Team members involved in each stage
"""
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(
            role=role,
            team_size=team_size,
            timeline=timeline
        )
        return result.strip()
    except Exception as e:
        return f"Error planning interview process: {str(e)}"


@tool
def generate_evaluation_criteria(input_str: str) -> str:
    """
    Create candidate evaluation criteria. Input is a JSON string with
    { "role": "", "key_skills": "", "company_values": "" }
    """
    try:
        data = json.loads(input_str)
        role = data.get("role", "")
        key_skills = data.get("key_skills", "")
        company_values = data.get("company_values", "")

        prompt = PromptTemplate(
            input_variables=["role", "key_skills", "company_values"],
            template="""
Create evaluation criteria for candidates applying for the {role} position.

Key Skills: {key_skills}
Company Values: {company_values}

Generate detailed evaluation criteria including:
1. Technical skills assessment
2. Soft skills assessment
3. Cultural fit indicators
4. Scoring rubric (1-5 scale)
5. Red flags and green flags to watch for
"""
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(
            role=role,
            key_skills=key_skills,
            company_values=company_values
        )
        return result.strip()
    except Exception as e:
        return f"Error generating evaluation criteria: {str(e)}"


@tool
def generate_hiring_analytics(input_str: str) -> str:
    """
    Generate hiring analytics and insights. Input is a JSON string with
    { "role": "", "applicants": "", "timeframe": "" }
    """
    try:
        data = json.loads(input_str)
        role = data.get("role", "")
        applicants = data.get("applicants", "")
        timeframe = data.get("timeframe", "")

        prompt = PromptTemplate(
            input_variables=["role", "applicants", "timeframe"],
            template="""
Generate hiring analytics and insights for the {role} position.

Number of Applicants: {applicants}
Timeframe: {timeframe}

Generate analytics including:
1. Expected conversion rates at each stage
2. Estimated time-to-hire
3. Recommended sourcing channels
4. Potential bottlenecks
5. Benchmark comparisons for similar roles
"""
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(
            role=role,
            applicants=applicants,
            timeframe=timeframe
        )
        return result.strip()
    except Exception as e:
        return f"Error generating hiring analytics: {str(e)}"

tools = [
    generate_job_description,
    plan_interview_process,
    generate_evaluation_criteria,
    generate_hiring_analytics
]


agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS, 
    verbose=True,
    memory=memory
)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('message', '')
        if 'chat_history' not in session:
            session['chat_history'] = []
        session['chat_history'].append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        response = agent_executor.run(user_input)
        agent_output = response if response else "I'm sorry, I couldn't process that."
        session['chat_history'].append({
            'role': 'assistant',
            'content': agent_output,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        return jsonify({
            'response': agent_output,
            'history': session['chat_history']
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    return jsonify({'history': session.get('chat_history', [])})


@app.route('/api/reset', methods=['POST'])
def reset_conversation():
    session['chat_history'] = []
    memory.clear()
    memory.chat_memory.add_message(system_message)

    return jsonify({
        'status': 'success',
        'message': 'Conversation reset successfully'
    })


if __name__ == '__main__':
    app.run(debug=True)


