import { cosineDistance, desc, gt, sql } from "drizzle-orm"

import { db } from "./db-config"
import { documents } from "./db-schema"
import { generateEmbedding } from "./embeddings"

export async function searchDocuments(
  query: string,
  limit: number = 5,
  threshold: number = 0.5
) {
  // If we have a Python backend configured, use it for Hybrid Search + Reranking
  if (process.env.PYTHON_BACKEND_URL) {
    try {
      const response = await fetch(`${process.env.PYTHON_BACKEND_URL}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      })
      if (response.ok) {
        const data = await response.json()
        return data.results.map((r: any) => ({
          id: r.id,
          content: r.content,
          similarity: r.score
        }))
      }
    } catch (error) {
      console.error("Failed to fetch from Python backend, falling back to local search:", error)
    }
  }

  const embedding = await generateEmbedding(query)

  const similarity = sql<number>`1-(${cosineDistance(documents.embedding, embedding)})`

  const similarDocuments = await db.select({
    id: documents.id,
    content: documents.content,
    similarity
  })
    .from(documents)
    .where(gt(similarity, threshold))
    .orderBy(desc(similarity))
    .limit(limit)

  return similarDocuments
}