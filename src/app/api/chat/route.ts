import { convertToModelMessages, streamText, UIMessage } from "ai";
import { groq } from "@ai-sdk/groq";

export async function POST(req: Request) {
  try {
    const { messages }: { messages: UIMessage[] } = await req.json();

    const result = streamText({
      model: groq("llama-3.3-70b-versatile"),
      messages: await convertToModelMessages(messages),
    });

    return result.toUIMessageStreamResponse();
  } catch (error) {
    console.error("error streaming chat completion", error);
    return new Response("failed to stream chat completion", { status: 500 });
  }
}
