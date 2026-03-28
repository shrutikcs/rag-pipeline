"use client";

import { Fragment, useState } from "react";
import { useChat } from "@ai-sdk/react";
import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import {
  Message,
  MessageContent,
  MessageResponse,
} from "@/components/ai-elements/message";
import {
  PromptInput,
  PromptInputBody,
  type PromptInputMessage,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputTools,
} from "@/components/ai-elements/prompt-input";
import { Spinner } from "@/components/ui/spinner";

const Chat = () => {
  const [input, setInput] = useState("");
  const { messages, sendMessage, status } = useChat();

  const handleSubmit = (message: PromptInputMessage) => {
    if (!message.text) {
      return;
    }

    sendMessage({ text: message.text });
    setInput("");
  };

  return (
    <div className="flex flex-col h-[calc(100dvh-57px)] max-w-3xl mx-auto p-4 md:px-8 md:py-6 w-full overflow-hidden">
      <Conversation className="flex-1 min-h-0 overflow-y-auto w-full minimal-scrollbar">
        <ConversationContent>
          {messages.map((message) => {
            return (
              <div key={message.id}>
                {message.parts.map((part, i) => {
                  switch (part.type) {
                    case "text":
                      return (
                        <Fragment key={`${message.id}-${i}`}>
                          <Message from={message.role}>
                            <MessageContent>
                              <MessageResponse>{part.text}</MessageResponse>
                            </MessageContent>
                          </Message>
                        </Fragment>
                      );

                    default:
                      return null;
                  }
                })}
              </div>
            );
          })}
          {(status === "submitted" || status === "streaming") && <Spinner />}
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>

      <div className="mt-4 w-full">
        <PromptInput onSubmit={handleSubmit}>
          <PromptInputBody>
            <PromptInputTextarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Send a message..."
              className="min-h-10 py-3"
            />
          </PromptInputBody>
          <PromptInputTools>{/* tools placeholder */}</PromptInputTools>
          <PromptInputSubmit />
        </PromptInput>
      </div>
    </div>
  );
};

export default Chat;
