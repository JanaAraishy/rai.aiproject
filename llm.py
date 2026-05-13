from groq import Groq


client = Groq(
    api_key="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"
)


def ask_llm(prompt, max_tokens=300):

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role": "system",
                "content": "أنت محلل أعمال محترف ومتخصص في تحليل آراء المستخدمين."
            },

            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.3,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content