describe('MediMaven Medical RAG Assistant', () => {
  beforeEach(() => {
    cy.visit('/')
  })

  it('completes full medical query flow', () => {
    // Welcome page
    cy.contains('MediMaven').should('be.visible')
    cy.contains(/medical AI assistant/i).should('be.visible')
    
    // Navigate to chat
    cy.contains('button', /start chat|get started/i).click()
    
    // Chat interface
    cy.url().should('include', '/chat')
    cy.get('textarea, input[type="text"]').should('be.visible')
    
    // Ask medical question
    cy.get('textarea, input[type="text"]')
      .type('What are the common treatments for type 2 diabetes?')
    
    cy.get('button[type="submit"]').click()
    
    // Wait for response
    cy.contains('Thinking', { timeout: 10000 }).should('be.visible')
    
    // Verify response quality
    cy.contains(/metformin|insulin|lifestyle/i, { timeout: 30000 })
      .should('be.visible')
    
    // Check citations
    cy.get('sup').contains('[1]').should('exist')
    
    // Follow-up question
    cy.get('textarea, input[type="text"]')
      .type('What about side effects of metformin?')
    cy.get('button[type="submit"]').click()
    
    // Verify contextual response
    cy.contains(/gastrointestinal|nausea|b12/i, { timeout: 30000 })
      .should('be.visible')
  })

  it('handles document citations properly', () => {
    cy.visit('/chat')
    
    // Ask for specific medical info
    cy.get('textarea, input[type="text"]')
      .type('Latest FDA guidelines on blood pressure medications')
    cy.get('button[type="submit"]').click()
    
    // Check citation hover
    cy.get('sup').contains('[1]').first().trigger('mouseenter')
    cy.contains(/source|medical/i).should('be.visible')
  })

  it('maintains conversation context', () => {
    cy.visit('/chat')
    
    // First message
    cy.get('textarea, input[type="text"]').type('I have a patient with hypertension')
    cy.get('button[type="submit"]').click()
    
    cy.contains(/blood pressure|hypertension/i, { timeout: 20000 })
    
    // Follow-up with context
    cy.get('textarea, input[type="text"]').type('What if they also have diabetes?')
    cy.get('button[type="submit"]').click()
    
    // Should maintain context
    cy.contains(/ACE inhibitors|ARBs|diabetic.*hypertension/i, { timeout: 20000 })
      .should('be.visible')
  })

  it('handles errors gracefully', () => {
    cy.visit('/chat')
    
    // Intercept API to force error
    cy.intercept('POST', '**/chat', { statusCode: 500 })
    
    cy.get('textarea, input[type="text"]').type('Test query')
    cy.get('button[type="submit"]').click()
    
    // Should show error state
    cy.contains(/error|retry|something went wrong/i).should('be.visible')
    cy.contains('button', 'Retry').should('be.visible')
  })
})
