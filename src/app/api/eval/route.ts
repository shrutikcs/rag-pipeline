import { NextResponse } from "next/server";
import { runRagAssessment } from "@/lib/eval";
import { runBatchEval, testDataset } from "@/lib/eval-dataset";

// ─── POST /api/eval ──────────────────────────────────────────────────────────
// Single-shot evaluation for live chat integration.
// Body: { query, context, retrievedChunks, answer, groundTruth? }

export async function POST(req: Request) {
  try {
    const { query, context, retrievedChunks, answer, groundTruth } =
      await req.json();

    if (!query || !context || !answer || !Array.isArray(retrievedChunks)) {
      return NextResponse.json(
        {
          error:
            "Missing required fields: query, context, retrievedChunks (array), answer",
        },
        { status: 400 }
      );
    }

    const assessment = await runRagAssessment({
      query,
      context,
      retrievedChunks,
      answer,
      groundTruth,
    });

    return NextResponse.json(assessment);
  } catch (error) {
    console.error("Error in POST /api/eval:", error);
    return NextResponse.json(
      {
        error: "Failed to run assessment",
        details: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
}

// ─── GET /api/eval ───────────────────────────────────────────────────────────
// Batch evaluation against the fixed XAUUSD/gold test dataset.
// Returns aggregate scores across all 4 metrics + per-item results.
//
// Example: curl http://localhost:3000/api/eval

export async function GET() {
  try {
    console.log(
      `[eval] Starting batch evaluation over ${testDataset.length} questions…`
    );
    const result = await runBatchEval(testDataset);
    console.log("[eval] Batch complete. Aggregates:", result.aggregates);
    return NextResponse.json(result);
  } catch (error) {
    console.error("Error in GET /api/eval:", error);
    return NextResponse.json(
      {
        error: "Failed to run batch evaluation",
        details: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
}
