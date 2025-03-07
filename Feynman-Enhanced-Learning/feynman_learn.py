import os
import re
import json
import streamlit as st
from typing import Dict, List, Optional

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_models import ChatTongyi
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv(
    "OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# Initialize tools and models
tavily_search = TavilySearchResults(max_results=3)
llm = ChatTongyi(
    model="qwq-plus",
    temperature=0,
    streaming=True,
)
embeddings = OpenAIEmbeddings(model="qwen2.5-7b-instruct-1m")


# Pydantic models (keep your existing models)
class LearningCheckpoint(BaseModel):
    """Structure for a single checkpoint"""

    description: str = Field(..., description="Main checkpoint description")
    criteria: List[str] = Field(..., description="List of success criteria")
    verification: str = Field(..., description="How to verify this checkpoint")


class Checkpoints(BaseModel):
    """Main checkpoints container with index tracking"""

    checkpoints: List[LearningCheckpoint] = Field(
        ...,
        description="List of checkpoints covering foundation, application, and mastery levels",
    )


# System messages (keep your existing prompts)
learning_checkpoints_generator = SystemMessage(
    content="""You will be given a learning topic title and learning objectives.
Your goal is to generate clear learning checkpoints that will help verify understanding and progress through the topic.
The output should be in JSON format with the following structure:
{
  "checkpoints": [
    {
      "description": "Level 1 checkpoint description",
      "criteria": ["criterion 1", "criterion 2", "criterion 3"],
      "verification": "How to verify this checkpoint"
    },
    {
      "description": "Level 2 checkpoint description", 
      "criteria": ["criterion 1", "criterion 2", "criterion 3"],
      "verification": "How to verify this checkpoint"
    },
    {
      "description": "Level 3 checkpoint description",
      "criteria": ["criterion 1", "criterion 2", "criterion 3"],
      "verification": "How to verify this checkpoint"
    }
  ]
}

IMPORTANT: Make sure "criteria" is formatted as a JSON array of strings, NOT as a single string.
Each criterion should be a separate string in the array, surrounded by quotes and separated by commas.

Requirements for each checkpoint:
- Description should be clear and concise
- Criteria should be specific and measurable (3-5 items)
- Verification method should be practical and appropriate for the level
- Verification will be checked by language model, so it must by in natural language
- All elements should align with the learning objectives
- Use action verbs and clear language
Ensure all checkpoints progress logically from foundation to mastery.
IMPORTANT - ANSWER WITH EXACTLY 3 CHECKPOINTS IN THE JSON FORMAT SHOWN ABOVE"""
)

question_generator = SystemMessage(
    content="""You will be given a checkpoint description, success criteria, and verification method.
Your goal is to generate an appropriate question that aligns with the checkpoint's verification requirements.
The question should:
1. Follow the specified verification method
2. Cover all success criteria
3. Encourage demonstration of understanding
4. Be clear and specific
Output should be a single, well-formulated question that effectively tests the checkpoint's learning objectives."""
)

answer_verifier = SystemMessage(
    content="""You will be given a student's answer, question, checkpoint details, and relevant context.
Your goal is to analyze the answer against the checkpoint criteria and provided context.
Analyze considering:
1. Alignment with verification method specified
2. Coverage of all success criteria
3. Use of relevant concepts from context
4. Depth and accuracy of understanding
Output should include:
- understanding_level: float between 0 and 1
- feedback: detailed explanation of the assessment
- suggestions: list of specific improvements
- context_alignment: boolean indicating if the answer aligns with provided context
The output should be in JSON format with the following structure:
{
  "understanding_level": 0.75,
  "feedback": "Detailed feedback on the answer",    
  "suggestions": ["Specific suggestions for improvement"],
  "context_alignment": true
}
"""
)

feynman_teacher = SystemMessage(
    content="""You will be given verification results, checkpoint criteria, and learning context.
Your goal is to create a Feynman-style teaching explanation for concepts that need reinforcement.
The explanation should include:
1. Simplified explanation without technical jargon
2. Concrete, relatable analogies
3. Key concepts to remember
Output should follow the Feynman technique:
- simplified_explanation: clear, jargon-free explanation
- key_concepts: list of essential points
- analogies: list of relevant, concrete comparisons
Focus on making complex ideas accessible and memorable.
The output should be in JSON format with the following structure:
{
    "simplified_explanation": "Simple explanation of the concept",
    "key_concepts": ["Key point 1", "Key point 2", "Key point 3"],
    "analogies": ["Analogy 1", "Analogy 2", "Analogy 3"]
}
"""
)


# Helper functions
def generate_checkpoints(topic: str, goals: str):
    """Creates learning checkpoints based on given topic and goals."""
    content = f"Topic {topic}\n\n Goals: {goals}"
    # content + learning_checkpoints_generator
    content = f"{learning_checkpoints_generator.content}\n{content}\n"
    messages = [
        SystemMessage(content=content),
    ]

    json_response = llm.invoke(messages)

    # Extract JSON content from response
    json_match = re.search(
        r"```(?:json)?\s*(.*?)\s*```", json_response.content, re.DOTALL
    )
    if json_match:
        json_str = json_match.group(1)
    else:
        json_str = json_response.content

    try:
        # Parse the JSON
        parsed_json = json.loads(json_str)

        # Create and validate the Checkpoints object
        checkpoints_list = []
        for checkpoint in parsed_json.get("checkpoints", []):
            # Ensure criteria is a list
            criteria = checkpoint.get("criteria", [])
            if isinstance(criteria, str):
                criteria = [criteria]

            checkpoints_list.append(
                LearningCheckpoint(
                    description=checkpoint.get("description", ""),
                    criteria=criteria,
                    verification=checkpoint.get("verification", ""),
                )
            )

        return Checkpoints(checkpoints=checkpoints_list)
    except Exception as e:
        st.error(f"Error parsing checkpoints: {e}")
        return create_fallback_checkpoints(topic)


def create_fallback_checkpoints(topic: str) -> Checkpoints:
    """Creates fallback checkpoints in case parsing fails."""
    return Checkpoints(
        checkpoints=[
            LearningCheckpoint(
                description=f"Understand basic concepts of {topic}",
                criteria=[
                    "Define key terms",
                    "Identify major types",
                    "Recognize common symptoms",
                ],
                verification=f"Explain the main concepts of {topic} in your own words",
            ),
            LearningCheckpoint(
                description=f"Apply diagnostic knowledge of {topic}",
                criteria=[
                    "Interpret test results",
                    "Identify indicators",
                    "Determine diagnostic steps",
                ],
                verification=f"Analyze a case study and identify key diagnostic findings",
            ),
            LearningCheckpoint(
                description=f"Develop management plans for {topic}",
                criteria=[
                    "Create differential diagnosis",
                    "Propose treatment options",
                    "Plan follow-up care",
                ],
                verification=f"Develop a complete management plan for a patient case",
            ),
        ]
    )


def search_web(query: str):
    """Performs web search for a given query."""
    try:
        return tavily_search.invoke(query)
    except Exception as e:
        st.error(f"Error during web search: {e}")
        return []


def generate_question(checkpoint: LearningCheckpoint):
    """Generates a verification question based on a checkpoint."""
    messages = [
        question_generator,
        SystemMessage(content=f"Checkpoint Description: {checkpoint.description}"),
        SystemMessage(content=f"Success Criteria: {', '.join(checkpoint.criteria)}"),
        SystemMessage(content=f"Verification Method: {checkpoint.verification}"),
    ]
    # 将SystemMessage合并为一个SystemMessage
    messages = [SystemMessage(content="\n".join([m.content for m in messages]))]
    response = llm.invoke(messages)
    return response.content


def verify_answer(
    checkpoint: LearningCheckpoint, question: str, answer: str, context: str
):
    """Verifies a student's answer against checkpoint criteria and context."""
    messages = [
        answer_verifier,
        SystemMessage(content=f"Question: {question}"),
        SystemMessage(content=f"Student Answer: {answer}"),
        SystemMessage(content=f"Checkpoint Description: {checkpoint.description}"),
        SystemMessage(content=f"Success Criteria: {', '.join(checkpoint.criteria)}"),
        SystemMessage(content=f"Verification Method: {checkpoint.verification}"),
        SystemMessage(content=f"Context: {context}"),
    ]
    messages = [SystemMessage(content="\n".join([m.content for m in messages]))]

    try:
        response = llm.invoke(
            messages,
            response_format={"type": "json_object"},
        )
        json_match = re.search(
            r"```(?:json)?\s*(.*?)\s*```", response.content, re.DOTALL
        )
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response.content

        result = json.loads(json_str)
        return result
    except Exception as e:
        st.error(f"Error verifying answer: {e}")
        return {
            "understanding_level": 0.5,
            "feedback": "Unable to properly assess your answer due to a system error.",
            "suggestions": ["Please try again later."],
            "context_alignment": False,
        }


def generate_teaching(
    checkpoint: LearningCheckpoint, verification_result: dict, context: str
):
    """Generates a Feynman-style teaching explanation."""
    messages = [
        feynman_teacher,
        SystemMessage(
            content=f"Verification Results: {json.dumps(verification_result)}"
        ),
        SystemMessage(content=f"Checkpoint Description: {checkpoint.description}"),
        SystemMessage(content=f"Success Criteria: {', '.join(checkpoint.criteria)}"),
        SystemMessage(content=f"Learning Context: {context}"),
    ]
    messages = [SystemMessage(content="\n".join([m.content for m in messages]))]

    try:
        response = llm.invoke(messages, response_format={"type": "json_object"})
        json_match = re.search(
            r"```(?:json)?\s*(.*?)\s*```", response.content, re.DOTALL
        )
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response.content

        result = json.loads(json_str)
        return result
    except Exception as e:
        st.error(f"Error generating teaching: {e}")
        return {
            "simplified_explanation": "We're having trouble generating a simplified explanation right now.",
            "key_concepts": [
                "Key concepts will be available when the system is functioning properly."
            ],
            "analogies": [
                "Analogies will be available when the system is functioning properly."
            ],
        }


# Streamlit UI
def main():
    st.set_page_config(page_title="Learning Assistant", page_icon="🎓", layout="wide")
    st.title("🎓 Learning Assistant")

    # Initialize session state
    if "step" not in st.session_state:
        st.session_state.step = 0
    if "topic" not in st.session_state:
        st.session_state.topic = ""
    if "goals" not in st.session_state:
        st.session_state.goals = ""
    if "checkpoints" not in st.session_state:
        st.session_state.checkpoints = None
    if "current_checkpoint" not in st.session_state:
        st.session_state.current_checkpoint = 0
    if "question" not in st.session_state:
        st.session_state.question = ""
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "verification_result" not in st.session_state:
        st.session_state.verification_result = None
    if "teaching" not in st.session_state:
        st.session_state.teaching = None

    # Step 0: Topic and Goals input
    if st.session_state.step == 0:
        st.subheader("Step 1: Define your learning topic and goals")

        st.session_state.topic = st.text_input(
            "What topic would you like to learn about?", st.session_state.topic
        )
        st.session_state.goals = st.text_area(
            "What are your learning goals?", st.session_state.goals
        )

        if (
            st.button("Generate Learning Checkpoints")
            and st.session_state.topic
            and st.session_state.goals
        ):
            with st.spinner("Generating learning checkpoints..."):
                st.session_state.checkpoints = generate_checkpoints(
                    st.session_state.topic, st.session_state.goals
                )
                st.session_state.step = 1

    # Step 1: Display checkpoints and begin assessment
    elif st.session_state.step == 1:
        st.subheader("Step 2: Learning Checkpoints")

        checkpoints = st.session_state.checkpoints

        # Display all checkpoints
        for i, cp in enumerate(checkpoints.checkpoints):
            with st.expander(
                f"Checkpoint {i+1}: {cp.description}",
                expanded=i == st.session_state.current_checkpoint,
            ):
                st.write("**Success Criteria:**")
                for criterion in cp.criteria:
                    st.write(f"- {criterion}")
                st.write("**Verification Method:**")
                st.write(cp.verification)

        if st.button("Begin Assessment"):
            # Generate question for current checkpoint
            current_cp = checkpoints.checkpoints[st.session_state.current_checkpoint]

            # Search for relevant content
            with st.spinner("Searching for relevant learning materials..."):
                search_query = f"{st.session_state.topic} {current_cp.description}"
                st.session_state.search_results = search_web(search_query)

            # Generate a verification question
            with st.spinner("Generating assessment question..."):
                st.session_state.question = generate_question(current_cp)

            st.session_state.step = 2

    # Step 2: Answer question and get verification
    elif st.session_state.step == 2:
        current_cp = st.session_state.checkpoints.checkpoints[
            st.session_state.current_checkpoint
        ]

        st.subheader(
            f"Checkpoint {st.session_state.current_checkpoint + 1}: Assessment"
        )
        st.write(f"**Question:** {st.session_state.question}")

        # Display relevant search results
        with st.expander("Learning Resources", expanded=False):
            for i, result in enumerate(st.session_state.search_results):
                st.write(f"**Source {i+1}:** {result['url']}")
                st.write(result["content"])
                st.write("---")

        # Get student answer
        answer = st.text_area("Your answer:", height=200)

        if st.button("Submit Answer") and answer:
            with st.spinner("Analyzing your answer..."):
                # Combine search results into context
                context = "\n\n".join(
                    [res["content"] for res in st.session_state.search_results]
                )

                # Verify the answer
                st.session_state.verification_result = verify_answer(
                    current_cp, st.session_state.question, answer, context
                )

                # Generate teaching if understanding level is below threshold
                if st.session_state.verification_result["understanding_level"] < 0.7:
                    st.session_state.teaching = generate_teaching(
                        current_cp, st.session_state.verification_result, context
                    )

                st.session_state.step = 3

    # Step 3: Display verification results and teaching
    elif st.session_state.step == 3:
        current_cp = st.session_state.checkpoints.checkpoints[
            st.session_state.current_checkpoint
        ]

        st.subheader(f"Checkpoint {st.session_state.current_checkpoint + 1}: Feedback")

        # Display verification results
        result = st.session_state.verification_result
        st.write("### Assessment Results")

        # Display understanding level with progress bar
        st.write("**Understanding Level:**")
        understanding = result.get("understanding_level", 0.5)
        st.progress(understanding)

        # Display feedback and suggestions
        st.write("**Feedback:**")
        st.write(result.get("feedback", "No feedback available"))

        st.write("**Suggestions for Improvement:**")
        for suggestion in result.get("suggestions", ["No suggestions available"]):
            st.write(f"- {suggestion}")

        # Display teaching if available
        if st.session_state.teaching:
            st.write("### Simplified Explanation")
            st.write(
                st.session_state.teaching.get(
                    "simplified_explanation", "No explanation available"
                )
            )

            st.write("**Key Concepts:**")
            for concept in st.session_state.teaching.get(
                "key_concepts", ["No key concepts available"]
            ):
                st.write(f"- {concept}")

            st.write("**Helpful Analogies:**")
            for analogy in st.session_state.teaching.get(
                "analogies", ["No analogies available"]
            ):
                st.write(f"- {analogy}")

        # Navigation buttons
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Try Again"):
                st.session_state.step = 2

        with col2:
            if (
                st.session_state.current_checkpoint
                < len(st.session_state.checkpoints.checkpoints) - 1
            ):
                if st.button("Next Checkpoint"):
                    st.session_state.current_checkpoint += 1
                    st.session_state.question = ""
                    st.session_state.verification_result = None
                    st.session_state.teaching = None
                    st.session_state.step = 1
            else:
                if st.button("Start Over"):
                    # Reset everything except topic and goals
                    st.session_state.step = 0
                    st.session_state.checkpoints = None
                    st.session_state.current_checkpoint = 0
                    st.session_state.question = ""
                    st.session_state.search_results = []
                    st.session_state.verification_result = None
                    st.session_state.teaching = None


if __name__ == "__main__":
    main()

