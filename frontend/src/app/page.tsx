'use client'
import { useState } from 'react'

interface AnalysisResult {
  sentiment: number;
  confidence: number;
  probabilities: number[][];
}

export default function Home() {
  const [text, setText] = useState('')
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [loading, setLoading] = useState(false)

  const analyzeSentiment = async () => {
    try {
      setLoading(true)
      const response = await fetch('http://localhost:8000/classify/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      })
      const data = await response.json()
      setResult(data)
    } catch (error) {
      console.error('Error:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="min-h-screen p-8">
      <div className="max-w-2xl mx-auto space-y-6">
        <h1 className="text-2xl font-bold">감정 분석</h1>
        
        <textarea
          className="w-full p-4 border rounded-lg"
          rows={4}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="분석할 텍스트를 입력하세요..."
        />
        
        <button
          onClick={analyzeSentiment}
          disabled={loading || !text}
          className="px-4 py-2 bg-blue-500 text-white rounded-lg disabled:opacity-50"
        >
          {loading ? '분석 중...' : '분석하기'}
        </button>

        {result !== null && (
          <div className="p-4 bg-gray-100 rounded-lg space-y-2">
            <p>감정 분석 결과: {result.sentiment === 1 ? '긍정적' : '부정적'}</p>
            <p>신뢰도: {(result.confidence * 100).toFixed(2)}%</p>
            <p>확률 분포: {result.probabilities[0].map(p => (p * 100).toFixed(2) + '%').join(', ')}</p>
          </div>
        )}
      </div>
    </main>
  )
}
