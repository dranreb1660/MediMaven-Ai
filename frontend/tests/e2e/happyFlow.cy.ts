describe('Happy path', () => {
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
        delay: 800 // Simulate response time
      })
    })
  })

  it('welcome → two queries → new chat', () => {
    cy.visit('/')
    cy.get('input[placeholder="Ask your health question..."]').type('What is diabetes?')
    cy.get('button[aria-label="Send message"]').click()
    cy.url().should('include','/chat')
    cy.contains(/diabetes|Mock response/i, {timeout:12_000}).should('be.visible')

    cy.get('textarea[placeholder="Ask your health question..."]').type('is it curable?')
    cy.get('button[aria-label="Send message"]').click()
    cy.contains(/curable|Mock response/i, {timeout:12_000}).should('be.visible')

    cy.get('button[title="New chat"]').click()
    cy.contains('medical assistant', {timeout:2_000}).should('be.visible')
  })
})
