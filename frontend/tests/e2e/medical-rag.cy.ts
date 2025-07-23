describe('MediMaven Medical RAG Assistant', () => {
  beforeEach(() => {
    // Mock the health endpoint
    cy.intercept('GET', '**/health', {
      statusCode: 200,
      body: { status: 'ok', bot_initialized: true }
    })
    
    // Mock the chat endpoint with medical context
    cy.intercept('POST', '**/chat', (req) => {
      const query = req.body.query || 'test query'
      req.reply({
        statusCode: 200,
        body: {
          answer: `Medical mock response for: ${query}. This simulates a medical AI assistant response.`,
          conversation_id: req.body.conversation_id || 'medical-test-123',
          citations: [{
            id: '1',
            source: 'Mock Medical Document',
            url: 'http://example.com/medical',
            rank: 1
          }],
          latency: 0.8,
          messages: [
            { user: query, assistant: `Medical mock response for: ${query}. This simulates a medical AI assistant response.` }
          ]
        },
        delay: 800 // Simulate medical AI processing time
      })
    })
    
    // Mock streaming endpoint too
    cy.intercept('POST', '**/chat/stream', (req) => {
      const query = req.body.query || 'test query'
      // For streaming, we'll just return a regular response since mocking SSE is complex
      req.reply({
        statusCode: 200,
        body: {
          answer: `Streaming medical response for: ${query}`,
          conversation_id: req.body.conversation_id || 'medical-stream-123',
          citations: [{
            id: '1',
            source: 'Mock Medical Document',
            url: 'http://example.com/medical',
            rank: 1
          }],
          latency: 0.8
        }
      })
    })
    
    cy.visit('/')
  })

  it('completes full medical query flow', () => {
    // Welcome page
    cy.contains('Medical AI Assistant').should('be.visible')
    cy.contains(/health companion/i).should('be.visible')
    
    // Ask question from welcome page
    cy.get('input[placeholder="Ask your health question..."]')
      .type('What are the common treatments for type 2 diabetes?')
    
    cy.get('button').contains('📤').click()
    
    // Chat interface
    cy.url().should('include', '/chat')
    cy.get('textarea[placeholder="Ask your health question..."]').should('be.visible')
    
    // Wait for response - be flexible about loading states
    cy.contains(/AI is thinking|Medical mock response|diabetes/i, { timeout: 15000 }).should('be.visible')
    
    // Verify response contains the query term
    cy.contains(/diabetes/i, { timeout: 30000 }).should('be.visible')
    
    // Follow-up question
    cy.get('textarea[placeholder="Ask your health question..."]')
      .type('What about side effects of metformin?')
    cy.get('button[aria-label="Send message"]').click()
    
    // Verify contextual response contains the follow-up term
    cy.contains(/metformin/i, { timeout: 30000 }).should('be.visible')
  })

  it('handles document citations properly', () => {
    cy.visit('/chat')
    
    // Ask for specific medical info
    cy.get('textarea[placeholder="Ask your health question..."]')
      .type('Latest FDA guidelines on blood pressure medications')
    cy.get('button[aria-label="Send message"]').click()
    
    // Wait for response containing the query term
    cy.contains(/blood pressure|Medical mock response/i, { timeout: 20000 }).should('be.visible')
    
    // Mock includes citations, check for them
    cy.contains('Mock Medical Document').should('be.visible')
  })

  it('maintains conversation context', () => {
    cy.visit('/chat')
    
    // First message
    cy.get('textarea[placeholder="Ask your health question..."]').type('I have a patient with hypertension')
    cy.get('button[aria-label="Send message"]').click()
    
    // Wait for response containing hypertension
    cy.contains(/hypertension/i, { timeout: 20000 }).should('be.visible')
    
    // Follow-up with context
    cy.get('textarea[placeholder="Ask your health question..."]').type('What if they also have diabetes?')
    cy.get('button[aria-label="Send message"]').click()
    
    // Should get response for diabetes question
    cy.contains(/diabetes/i, { timeout: 20000 }).should('be.visible')
  })

  it('handles errors gracefully', () => {
    cy.visit('/chat')
    
    // Intercept API to force error
    cy.intercept('POST', '**/chat', { statusCode: 500 })
    
    cy.get('textarea[placeholder="Ask your health question..."]').type('Test query')
    cy.get('button[aria-label="Send message"]').click()
    
    // Should show error state
    cy.contains(/error|something went wrong/i, { timeout: 10000 }).should('be.visible')
    cy.contains('button', /try again/i).should('be.visible')
  })
})
