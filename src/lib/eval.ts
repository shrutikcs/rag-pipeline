import { generateText, Output } from "ai";
import { groq } from "@ai-sdk/groq";
import { z } from "zod";
import type { LanguageModel } from "ai";

// ─── Models ──────────────────────────────────────────────────────────────────
// Two models to distribute token load and stay within rate limits.
// - FAST: lighter metrics that need less reasoning (faithfulness, answer relevance)
// - SMART: heavier reasoning metrics (context precision, context recall)

const MODEL_FAST: LanguageModel = groq("openai/gpt-oss-20b");
const MODEL_SMART: LanguageModel = groq("openai/gpt-oss-120b");

// ─── Shared schema ───────────────────────────────────────────────────────────

const evalSchema = z.object({
  score: z.number().min(0).max(1),
});

type EvalResult = z.infer<typeof evalSchema>;

async function runEval(
  model: LanguageModel,
  system: string,
  prompt: string
): Promise<EvalResult> {
  const result = await generateText({
    model,
    output: Output.object({ schema: evalSchema }),
    system,
    prompt,
  });
  return result.output;
}

// ─── Metric 1: Faithfulness ──────────────────────────────────────────────────
// Is every claim in the answer supported by the retrieved context?

export async function evaluateFaithfulness(
  query: string,
  answer: string,
  context: string
): Promise<EvalResult> {
  return runEval(
    MODEL_FAST,
    "You are a RAG evaluator. Score FAITHFULNESS: are all answer claims supported by the context? Reply with JSON {score: 0-1}.",
    `QUERY: ${query}\nCONTEXT: ${context}\nANSWER: ${answer}`
  );
}

// ─── Metric 2: Answer Relevance ───────────────────────────────────────────────
// Does the answer actually address the user's query?

export async function evaluateAnswerRelevance(
  query: string,
  answer: string,
  context: string
): Promise<EvalResult> {
  return runEval(
    MODEL_FAST,
    "You are a RAG evaluator. Score ANSWER RELEVANCE: does the answer directly address the query? Reply with JSON {score: 0-1}.",
    `QUERY: ${query}\nANSWER: ${answer}`
  );
}

// ─── Metric 3: Context Precision ─────────────────────────────────────────────
// Are the retrieved chunks actually relevant to the query?
// Low precision = lots of noise retrieved alongside useful chunks.

export async function evaluateContextPrecision(
  query: string,
  retrievedChunks: string[]
): Promise<EvalResult> {
  const numberedChunks = retrievedChunks
    .map((c, i) => `[${i + 1}] ${c}`)
    .join("\n\n");

  return runEval(
    MODEL_SMART,
    "You are a RAG evaluator. Score CONTEXT PRECISION: what fraction of the chunks are relevant to the query? Reply with JSON {score: 0-1}.",
    `QUERY: ${query}\nCHUNKS:\n${numberedChunks}`
  );
}

// ─── Metric 4: Context Recall ─────────────────────────────────────────────────
// Did we retrieve all the information needed to answer?
// Measured against a ground-truth answer.

export async function evaluateContextRecall(
  query: string,
  groundTruth: string,
  retrievedChunks: string[]
): Promise<EvalResult> {
  const numberedChunks = retrievedChunks
    .map((c, i) => `[${i + 1}] ${c}`)
    .join("\n\n");

  return runEval(
    MODEL_SMART,
    "You are a RAG evaluator. Score CONTEXT RECALL: how much of the ground-truth answer is supported by the retrieved chunks? Reply with JSON {score: 0-1}.",
    `QUERY: ${query}\nGROUND TRUTH: ${groundTruth}\nCHUNKS:\n${numberedChunks}`
  );
}

// ─── Combined runner ─────────────────────────────────────────────────────────

export interface RagAssessmentInput {
  query: string;
  context: string;           // pre-formatted string for faithfulness / relevance
  retrievedChunks: string[]; // raw chunks for precision / recall
  answer: string;
  groundTruth?: string;      // required for context recall
}

export interface RagAssessmentResult {
  faithfulness: EvalResult;
  answerRelevance: EvalResult;
  contextPrecision: EvalResult;
  contextRecall: EvalResult | null;
  averageScore: number;
}

export async function runRagAssessment(
  input: RagAssessmentInput
): Promise<RagAssessmentResult> {
  const { query, context, retrievedChunks, answer, groundTruth } = input;

  // Sequential — one call at a time to stay within rate limits
  const faithfulness = await evaluateFaithfulness(query, answer, context);
  const answerRelevance = await evaluateAnswerRelevance(query, answer, context);
  const contextPrecision = await evaluateContextPrecision(query, retrievedChunks);
  const contextRecall = groundTruth
    ? await evaluateContextRecall(query, groundTruth, retrievedChunks)
    : null;


  const scores = [
    faithfulness.score,
    answerRelevance.score,
    contextPrecision.score,
    ...(contextRecall ? [contextRecall.score] : []),
  ];
  const averageScore = scores.reduce((a, b) => a + b, 0) / scores.length;

  return {
    faithfulness,
    answerRelevance,
    contextPrecision,
    contextRecall,
    averageScore,
  };
}
