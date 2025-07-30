describe('MediMaven Chat', () => {
  beforeEach(() => {
    // Mock the health endpoint
    cy.intercept('GET', '**/health', {
      statusCode: 200,
      body: { status: 'ok', bot_initialized: true }
    })
    
    // Mock the chat endpoint for reliable CI testing
    cy.intercept('POST', '**/chat', (req) => {
      const query = req.body.query || 'test query'
      req.reply({
        statusCode: 200,
        body: {
          answer: `Mock response: ${query}`,
          conversation_id: 'test-conversation-123',
          citations: [{
            id: '1',
            source: 'Mock Medical Document',
            url: 'http://example.com',
            rank: 1
          }],
          latency: 0.5,
          messages: [
            { user: query, assistant: `Mock response: ${query}` }
          ]
        },
        delay: 1000 // Simulate response time
      })
    })
  })

  it('loads the app', () => {
    cy.visit('/')
    cy.contains('MediMaven AI Assistant').should('be.visible')
  })

  it('can send a message from welcome page', () => {
    cy.visit('/')
    cy.get('input[placeholder="Ask your health question..."]').type('What are symptoms of diabetes?')
    cy.get('button[aria-label="Send message"]').click()
    cy.url().should('include', '/chat')
    
    // Look for the query or response - more flexible
    cy.contains(/diabetes|Mock response/i, { timeout: 10000 }).should('be.visible')
  })

  it('can send a message from chat page', () => {
    cy.visit('/chat')
    cy.get('textarea[placeholder="Ask your health question..."]').should('be.visible')
    cy.get('textarea[placeholder="Ask your health question..."]').type('What are symptoms of diabetes?')
    cy.get('button[aria-label="Send message"]').click()
    
    // Wait for response - look for either thinking state or direct response
    cy.contains(/AI is thinking|Mock response|diabetes/i, { timeout: 15000 }).should('be.visible')
  })
})
