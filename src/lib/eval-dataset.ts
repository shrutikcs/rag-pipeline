import { generateText } from "ai";
import { groq } from "@ai-sdk/groq";
import { searchDocuments } from "./search";
import { runRagAssessment, RagAssessmentResult, getNextEvalModel } from "./eval";

// ─── Fixed test dataset ───────────────────────────────────────────────────────
// 18 gold/XAUUSD Q&A pairs covering the headline topics from the PDF.

export interface DatasetItem {
  query: string;
  groundTruth: string;
}

export const testDataset: DatasetItem[] = [
  {
    query: "How do US tariffs and trade wars affect XAUUSD?",
    groundTruth:
      "Trump's tariffs created massive uncertainty that drove safe-haven demand for gold. A 39% tariff on Swiss gold bar imports directly increased acquisition costs. Trade wars also weaken the dollar, boosting gold prices further. Gold surged past $3,000 when tariffs launched in April 2025.",
  },
  {
    query: "What is de-dollarization and how does it impact gold prices?",
    groundTruth:
      "De-dollarization is the global shift away from USD as the world's reserve currency. Central banks — especially China, Turkey, and Poland — are buying gold to reduce dollar exposure. This creates persistent structural buying pressure on gold regardless of macro conditions.",
  },
  {
    query: "How does the US national debt affect gold?",
    groundTruth:
      "Rising US national debt (now ~$38T) erodes confidence in the dollar and raises long-term inflation risk. Investors hedge against currency debasement by buying gold. Higher debt also forces more Treasury issuance, pushing up term premiums and weakening the dollar — both bullish for gold.",
  },
  {
    query: "How does the Federal Reserve's independence affect XAUUSD?",
    groundTruth:
      "Political pressure on the Fed (like Trump's attacks on its independence) signals possible monetary manipulation, which spooks markets. This uncertainty drives safe-haven flows into gold. A dovish or politically influenced Fed leadership transition is also bullish for gold.",
  },
  {
    query: "What drove gold past $4,000 and then $5,000 in 2025?",
    groundTruth:
      "Gold surged 60%+ in 2025 driven by a combination of factors: Trump tariff uncertainty, Fed rate cuts, aggressive central bank buying, ETF inflows, a weakening dollar, and geopolitical tensions in the Middle East and Ukraine. It hit $4,000 in October 2025 and $5,595 in early 2026.",
  },
  {
    query: "How do US government shutdowns affect gold prices?",
    groundTruth:
      "Government shutdowns increase uncertainty about fiscal stability and the US economy, which boosts safe-haven demand for gold. The 2025 shutdown was one of several catalysts that pushed gold toward $4,000.",
  },
  {
    query: "What is the Shanghai Gold Exchange (SGE) premium and what does it signal?",
    groundTruth:
      "The SGE premium is the difference between Chinese domestic gold prices and the international LBMA spot price. A high premium signals intense local demand and supply tightness in China. It's a leading indicator of physical demand pressure that can push global gold prices higher.",
  },
  {
    query: "How does gold perform when stock-bond correlations are high?",
    groundTruth:
      "When stocks and bonds fall together (positive correlation), the traditional 60/40 portfolio fails as a hedge. In this environment, gold becomes more valuable as a diversifier since its correlation to the dollar remains stable. High stock-bond correlations structurally increase gold demand.",
  },
  {
    query: "What role does jewelry demand play in gold price movements?",
    groundTruth:
      "Jewelry is a major source of physical gold demand, especially from India and China. However, at record-high gold prices (like $4,000+), jewelry demand drops sharply as it becomes unaffordable. This is a natural ceiling effect and can offset investment-driven buying.",
  },
  {
    query: "How did gold ETF flows change in 2025 and what does it mean for prices?",
    groundTruth:
      "Gold ETFs flipped from major outflows to strong inflows in 2025, with investment demand surging from 205T to 551T tonnes. Large ETF inflows signal institutional conviction in the gold bull case and directly increase market demand, supporting price appreciation.",
  },
  {
    query: "What is the impact of Japanese yen weakness and BoJ policy on gold?",
    groundTruth:
      "Yen weakness drives Japanese investors to seek alternative stores of value — gold being a top choice. BoJ policy normalization and rising JGB yields also risk unwinding global carry trades, creating volatility in leveraged positions including precious metals.",
  },
  {
    query: "How do China's institutional gold allocation policies affect XAUUSD?",
    groundTruth:
      "In 2025, China launched a pilot allowing 10 insurers to allocate up to 1% of assets to gold. If the PBoC expands these limits, it would unleash a massive wave of institutional demand. China also became a custodian for foreign sovereign gold reserves, further anchoring demand.",
  },
  {
    query: "What is the relationship between global debt levels and gold?",
    groundTruth:
      "Global sectoral debt reached $340 trillion in 2025, with government debt at a record 30% share. At 3-4x global GDP, this raises fears of currency debasement and inflation. Gold is increasingly used as a hedge against sovereign debt risk, supporting its long-term bull case.",
  },
  {
    query: "How does profit-taking affect short-term gold price movements?",
    groundTruth:
      "After parabolic rallies, gold is highly vulnerable to sharp corrections from profit-taking. In early 2026, gold dropped ~10% in a single session after hitting $5,595. Once selling starts, it snowballs. This is a key risk for short-term traders despite the long-term bullish trend.",
  },
  {
    query: "What does the World Gold Council's GRAM model say drives gold returns?",
    groundTruth:
      "The WGC Gold Return Attribution Model (GRAM) breaks gold drivers into four categories: economic expansion, risk & uncertainty, opportunity cost, and momentum. In 2025, geopolitical risk and USD weakness accounted for ~16 percentage points of gold's return, with momentum adding 9 points.",
  },
];

// ─── Per-item result ──────────────────────────────────────────────────────────

export interface BatchEvalItemResult {
  query: string;
  groundTruth: string;
  answer: string;
  retrievedChunks: string[];
  metrics: RagAssessmentResult;
}

// ─── Batch runner ─────────────────────────────────────────────────────────────

export interface BatchEvalResult {
  aggregates: {
    faithfulness: number;
    answerRelevance: number;
    contextPrecision: number;
    contextRecall: number;
    average: number;
  };
  results: BatchEvalItemResult[];
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function runSingleEval(item: DatasetItem): Promise<BatchEvalItemResult> {
  // 1. Retrieve context from the knowledge base
  const docs = await searchDocuments(item.query, 5, 0.3);
  const retrievedChunks = docs.map((d) => d.content);
  const context = retrievedChunks
    .map((c, i) => `[${i + 1}] ${c}`)
    .join("\n\n");

  // 2. Generate an answer (sequential — one call at a time per item)
  const { text: answer } = await generateText({
    model: getNextEvalModel(),
    system:
      "You are a helpful assistant. Answer the user's question based on the provided context. " +
      "Be concise and accurate.",
    prompt: `CONTEXT:\n${context || "No relevant context found."}\n\nQUESTION: ${item.query}`,
  });

  // 3. Run metrics sequentially to stay within per-model rate limits
  const metrics = await runRagAssessment({
    query: item.query,
    context,
    retrievedChunks,
    answer,
    groundTruth: item.groundTruth,
  });

  return {
    query: item.query,
    groundTruth: item.groundTruth,
    answer,
    retrievedChunks,
    metrics,
  };
}

function avg(nums: number[]) {
  return nums.length === 0
    ? 0
    : nums.reduce((a, b) => a + b, 0) / nums.length;
}

export async function runBatchEval(
  dataset: DatasetItem[] = testDataset
): Promise<BatchEvalResult> {
  const results: BatchEvalItemResult[] = [];
  const DELAY_MS = 2000; // 2s between items
  const CHUNK_SIZE = 2; // Process 3 items concurrently

  for (let i = 0; i < dataset.length; i += CHUNK_SIZE) {
    const chunk = dataset.slice(i, i + CHUNK_SIZE);
    console.log(`[eval] Processing items ${i + 1} to ${Math.min(i + CHUNK_SIZE, dataset.length)}/${dataset.length}…`);
    
    const chunkResults = await Promise.all(chunk.map((item) => runSingleEval(item)));
    results.push(...chunkResults);
    
    if (i + CHUNK_SIZE < dataset.length) await sleep(DELAY_MS);
  }

  const faithfulnessScores = results.map((r) => r.metrics.faithfulness.score);
  const relevanceScores = results.map((r) => r.metrics.answerRelevance.score);
  const precisionScores = results.map((r) => r.metrics.contextPrecision.score);
  const recallScores = results
    .filter((r) => r.metrics.contextRecall !== null)
    .map((r) => r.metrics.contextRecall!.score);

  const aggregates = {
    faithfulness: avg(faithfulnessScores),
    answerRelevance: avg(relevanceScores),
    contextPrecision: avg(precisionScores),
    contextRecall: avg(recallScores),
    average: avg([
      avg(faithfulnessScores),
      avg(relevanceScores),
      avg(precisionScores),
      avg(recallScores),
    ]),
  };

  return { aggregates, results };
}
