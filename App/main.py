from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr, Field, ValidationError
from typing import Optional
import os, json

from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain_together import ChatTogether

app = FastAPI(title="Ementora AI Assistant")

HISTORY_DIR = "chat_history"
os.makedirs(HISTORY_DIR, exist_ok=True)

# === User Info Schema ===
class UserInfo(BaseModel):
    name: Optional[str] = Field(default=None, max_length=50)
    email: Optional[EmailStr] = Field(default=None)
    phone: Optional[str] = Field(default=None, max_length=10, min_length=10)
    user_id: Optional[str] = Field(default=None)

# === LLM Integration ===
llm = ChatTogether(
    model="meta-llama/Llama-3-70b-chat-hf",
    temperature=0.7,
    max_tokens=512,
    api_key="1882eed17b84bbee420d26f95a1342d5453e16fb1adfbe9caf161e1136143d7f" 
)

# === Custom Prompt Template for Ementora ===
template = """
You are Ementora's AI Assistant — an intelligent, friendly, and informative chatbot.

If the user's question is not clearly related to Ementora's services, training programs, education, career guidance, or student support, respond with:
"Sorry, I don't know."
Do not attempt to answer irrelevant or unrelated questions, even if you know the answer.

These are Few FAQ's-

1. What is Ementora?
Ementora is an AI-powered EdTech startup, co-founded by IIT Kharagpur alumni and industry experts. It delivers immersive technical and management internship training, advanced dual global courses, and specialized academic support services.


2. What services does Ementora offer?
Ementora’s key offerings include:

AI-powered dual global academic programs

Campus-to-corporate global training

AI-based industry-immersed internships

Foreign language training

Entrepreneurship Development Program (EDP) & startup incubation

AI-generated project and assignment guidance

Exclusive career counseling & global admission assistance

Customized real-time virtual classes

AI-powered skill development programs

3. Who is Ementora designed for?
Ementora works with students from top colleges, universities, and institutes, especially those seeking internships, career-driven skill development, and global training experiences

4. How is AI integrated into Ementora’s offerings?
AI is deeply embedded across its programs—from personalized project guidance & immersive internships to skill development and advanced training initiatives—ensuring practical learning tailored to real-world needs.

5. How can one get in touch or apply?
Contact options listed:

Email: ementoraglobal@gmail.com

Phone: +91 9830419001

Office Address: 95, Purbayan, C.S. Road, Kolkata‑700105, India



Your purpose is to:
1. Greet and understand the user.
2. Help users with queries related to Ementora's services, mentoring, career guidance, and study support.
3. Provide helpful, accurate, and clear answers in a warm tone.

User Profile:
- Name: {name}
- Email: {email}
- Phone: {phone}
- ID: {user_id}

Conversation History:
{history}

User's Current Message:
{message}

Ementora Assistant's Reply:
"""

prompt = PromptTemplate(
    input_variables=["name", "email", "phone", "user_id", "history", "message"],
    template=template
)

FIELDS = ["name", "email", "phone", "user_id"]

# === Chat Schemas ===
class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str

# === Utility Functions ===
def load_user_data(uid: str):
    path = os.path.join(HISTORY_DIR, f"{uid}.json")
    if not os.path.exists(path):
        return {
            "user_info": {field: None for field in FIELDS},
            "conversation": []
        }
    with open(path, "r") as f:
        return json.load(f)

def save_user_data(uid: str, data):
    path = os.path.join(HISTORY_DIR, f"{uid}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# === Main Chat Endpoint ===
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        data = load_user_data(req.user_id)
        info = data["user_info"]
        conversation = data["conversation"]

        # Step 1: Collect missing profile fields
        for field in FIELDS:
            if not info.get(field):
                if conversation and conversation[-1]["type"] == "ai":
                    info[field] = req.message
                    try:
                        UserInfo(**info)
                    except ValidationError as ve:
                        msg = ve.errors()[0]["msg"]
                        reply = f"❌ Invalid {field}: {msg}. Please try again."
                        conversation.append({"type": "human", "content": req.message})
                        conversation.append({"type": "ai", "content": reply})
                        return ChatResponse(reply=reply)

                    conversation.append({"type": "human", "content": req.message})
                    if field != FIELDS[-1]:
                        next_field = FIELDS[FIELDS.index(field)+1]
                        reply = f"Thanks! May I also know your {next_field}?"
                    else:
                        reply = "✅ You're all set! Feel free to ask me anything about Ementora or your studies."
                    conversation.append({"type": "ai", "content": reply})
                    save_user_data(req.user_id, {"user_info": info, "conversation": conversation})
                    return ChatResponse(reply=reply)

                reply = f"Hi! Could you please share your {field}?"
                conversation.append({"type": "ai", "content": reply})
                save_user_data(req.user_id, {"user_info": info, "conversation": conversation})
                return ChatResponse(reply=reply)

        # Step 2: Answer user queries
        messages = [SystemMessage(content="You are a helpful AI assistant for Ementora.")]
        for msg in conversation:
            if msg["type"] == "human":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["type"] == "ai":
                messages.append(AIMessage(content=msg["content"]))

        formatted_prompt = prompt.format(
            name=info["name"],
            email=info["email"],
            phone=info["phone"],
            user_id=info["user_id"],
            history="\n".join([f"{'User' if m['type'] == 'human' else 'Bot'}: {m['content']}" for m in conversation]),
            message=req.message
        )

        messages.append(HumanMessage(content=req.message))
        response = llm.invoke(messages)

        conversation.append({"type": "human", "content": req.message})
        conversation.append({"type": "ai", "content": response.content})
        save_user_data(req.user_id, {"user_info": info, "conversation": conversation})

        return ChatResponse(reply=response.content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === Profile Fetch Endpoint ===
@app.get("/profile/{user_id}")
def get_profile(user_id: str):
    data = load_user_data(user_id)
    info = data.get("user_info", {})
    if all(info.values()):
        return info
    raise HTTPException(status_code=404, detail="Incomplete profile")
