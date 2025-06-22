// File: app/page.tsx (or pages/index.tsx if not using app dir)
'use client'

import { useState } from 'react'

export default function Home() {
  const [prompt, setPrompt] = useState('')
  const [messages, setMessages] = useState<{ role: string; content: string }[]>([])
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!prompt.trim()) return

    const userMessage = { role: 'user', content: prompt }
    setMessages((prev) => [...prev, userMessage])
    setPrompt('')
    setLoading(true)

    try {
      const res = await fetch('http://localhost:8000/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      })

      if (!res.ok) throw new Error('Server error')

      const data = await res.json()

      const botMessage = { role: 'assistant', content: data.final_response }
      setMessages((prev) => [...prev, botMessage])
      const metrics = data.metrics
      console.log("metrics: ", metrics)

    } catch (err) {
      const errorMessage = {
        role: 'assistant',
        content: '⚠️ Sorry, something went wrong while contacting the server.',
      }
      setMessages((prev) => [...prev, errorMessage])
      console.error(err)
    }

    setLoading(false)
  }


  return (
    <main className="max-w-2xl mx-auto p-6">
      <h1 className="text-2xl font-semibold mb-4">Cache-22 Chat</h1>

      <div className="border rounded p-4 h-[400px] overflow-y-auto mb-4 bg-gray-50">
        {messages.map((m, i) => (
          <div key={i} className={`mb-2 ${m.role === 'user' ? 'text-right' : 'text-left'}`}>
            <div className={`inline-block px-4 py-2 rounded-lg ${m.role === 'user' ? 'bg-blue-200' : 'bg-gray-300'}`}>
              {m.content}
            </div>
          </div>
        ))}
        {loading && <div className="text-gray-500">Thinking...</div>}
      </div>

      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          className="flex-1 border rounded px-4 py-2"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Ask me anything..."
        />
        <button className="bg-blue-500 text-white px-4 py-2 rounded" type="submit">
          Send
        </button>
      </form>
    </main>
  )
}
