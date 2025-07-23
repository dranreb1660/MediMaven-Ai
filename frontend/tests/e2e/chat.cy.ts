describe('MediMaven Chat', () => {
  it('loads the app', () => {
    cy.visit('/')
    cy.contains('Medical AI Assistant').should('be.visible')
  })

  it('can send a message from welcome page', () => {
    cy.visit('/')
    cy.get('input[placeholder="Ask your health question..."]').type('What are symptoms of diabetes?')
    cy.get('button').contains('📤').click()
    cy.url().should('include', '/chat')
    cy.contains('AI is thinking', { timeout: 10000 }).should('be.visible')
  })

  it('can send a message from chat page', () => {
    cy.visit('/chat')
    cy.get('textarea[placeholder="Ask your health question..."]').should('be.visible')
    cy.get('textarea[placeholder="Ask your health question..."]').type('What are symptoms of diabetes?')
    cy.get('button[aria-label="Send message"]').click()
    cy.contains('🤖 AI is thinking', { timeout: 10000 }).should('be.visible')
    cy.contains('mock response', { timeout: 15000 }).should('be.visible')
  })
})
