describe('MediMaven Medical RAG Assistant', () => {
  beforeEach(() => {
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
    
    // Wait for response
    cy.contains('🤖 AI is thinking', { timeout: 10000 }).should('be.visible')
    
    // Verify response (using mock response content)
    cy.contains(/mock response.*diabetes/i, { timeout: 30000 })
      .should('be.visible')
    
    // Follow-up question
    cy.get('textarea[placeholder="Ask your health question..."]')
      .type('What about side effects of metformin?')
    cy.get('button[aria-label="Send message"]').click()
    
    // Verify contextual response (mock will echo the question)
    cy.contains(/mock response.*metformin/i, { timeout: 30000 })
      .should('be.visible')
  })

  it('handles document citations properly', () => {
    cy.visit('/chat')
    
    // Ask for specific medical info
    cy.get('textarea[placeholder="Ask your health question..."]')
      .type('Latest FDA guidelines on blood pressure medications')
    cy.get('button[aria-label="Send message"]').click()
    
    // Wait for response first
    cy.contains(/mock response/i, { timeout: 20000 }).should('be.visible')
    
    // Mock backend includes citations, check for them
    cy.contains('Mock Medical Document').should('be.visible')
  })

  it('maintains conversation context', () => {
    cy.visit('/chat')
    
    // First message
    cy.get('textarea[placeholder="Ask your health question..."]').type('I have a patient with hypertension')
    cy.get('button[aria-label="Send message"]').click()
    
    cy.contains(/mock response.*hypertension/i, { timeout: 20000 })
    
    // Follow-up with context
    cy.get('textarea[placeholder="Ask your health question..."]').type('What if they also have diabetes?')
    cy.get('button[aria-label="Send message"]').click()
    
    // Should get response for diabetes question
    cy.contains(/mock response.*diabetes/i, { timeout: 20000 })
      .should('be.visible')
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
