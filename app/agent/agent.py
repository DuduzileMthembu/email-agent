import os

import asyncio
 
from dotenv import load_dotenv
 
from agents import (

    Agent,

    Runner,

    AsyncOpenAI,

    OpenAIChatCompletionsModel,

    set_tracing_disabled,

)
 
from agents.mcp import MCPServerStdio
 
 
# Load our .env file

load_dotenv()
 
 
# Get the Grok API key

xai_api_key = os.getenv("XAI_API_KEY")
 
if not xai_api_key:

    raise ValueError(

        "XAI_API_KEY is missing from the .env file."

    )
 
 
# Connect to the xAI API

xai_client = AsyncOpenAI(

    api_key=xai_api_key,

    base_url="https://api.x.ai/v1",

)
 
 
# Create the Grok model

grok_model = OpenAIChatCompletionsModel(

    model="grok-4.7",

    openai_client=xai_client,

)
 
 
# We are using Grok instead of OpenAI

set_tracing_disabled(True)
 
 
# Create the AI agent

email_agent = Agent(

    name="Email Assistant",
 
    instructions="""

    You are an intelligent email assistant.
 
    Your job is to help the user manage their Gmail inbox.
 
    Follow these steps:
 
    1. Use the read_emails tool to retrieve the user's emails.
 
    2. For every email:

       - Identify the sender.

       - Identify the subject.

       - Understand the main message.

       - Summarize the email.

       - Decide whether the sender expects a response.
 
    3. An email generally needs a response when:

       - The sender asks a question.

       - The sender requests information.

       - The sender asks the user to perform an action.

       - The sender expects confirmation or a decision.

       - A response would reasonably be expected in the conversation.
 
    4. An email generally does NOT need a response when:

       - It is a newsletter.

       - It is an automated notification.

       - It is purely informational.

       - No response or action is requested.
 
    5. If an email requires a response:

       - Write a professional and natural reply.

       - Use the information available in the original email.

       - Do not invent information.

       - Create a Gmail draft using the create_email_draft tool.

       - Do NOT send the email.
 
    6. If an email does not require a response:

       - Do not create a draft.
 
    7. At the end, provide the user with:

       - A summary of the emails.

       - Which emails require responses.

       - Which drafts were created.
 
    IMPORTANT:

    Never send an email.

    Only create drafts for the user to review.

    """,
 
    model=grok_model,

)
 
 
async def main():
 
    # Start the MCP server

    async with MCPServerStdio(

        name="Email MCP Server",
 
        params={

            "command": "uv",

            "args": [

                "run",

                "python",

                "app/mcp/email_server.py",

            ],

        },

    ) as server:
 
        # Give the AI agent access to our MCP tools

        email_agent.mcp_servers = [server]
 
        # Tell the agent what we want it to do

        result = await Runner.run(

            email_agent,

            """

            Check my latest 5 emails.
 
            Summarize each email.
 
            Determine which emails require a response.
 
            For emails that require a response,

            write an appropriate reply and create

            a Gmail draft.
 
            Do not send any emails.

            """,

        )
 
        print("\n")

        print(result.final_output)
 
 
if __name__ == "__main__":

    asyncio.run(main())
 