import { embed, embedMany } from "ai";
import { google, type GoogleEmbeddingModelOptions } from "@ai-sdk/google";

export async function generateEmbedding(text: string) {
  const input = text.replace("\n", " ");

  const { embedding } = await embed({
    model: google.embedding("gemini-embedding-2-preview"),
    value: input,
    providerOptions: {
      google: {
        outputDimensionality: 300,
      } satisfies GoogleEmbeddingModelOptions,
    },
  });

  return embedding;
}

export async function generateEmbeddings(texts: string[]) {
  const inputs = texts.map((text) => text.replace("\n", " "));

  const { embeddings } = await embedMany({
    model: google.embedding("gemini-embedding-2-preview"),
    values: inputs,
    providerOptions: {
      google: {
        outputDimensionality: 300,
      } satisfies GoogleEmbeddingModelOptions,
    },
  });
  return embeddings;
}
