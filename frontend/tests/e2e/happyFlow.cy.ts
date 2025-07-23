describe('Happy path', () => {
  it('welcome → two queries → new chat', () => {
    cy.visit('/')
    cy.get('input[placeholder="Ask your health question..."]').type('What is diabetes?')
    cy.get('button').contains('📤').click()
    cy.url().should('include','/chat')
    cy.contains('diabetes', {timeout:12_000}).should('be.visible')

    cy.get('textarea[placeholder="Ask your health question..."]').type('is it curable?')
    cy.get('button[aria-label="Send message"]').click()
    cy.contains('curable', {timeout:12_000}).should('be.visible')

    cy.contains('Clear chat').click()
    cy.contains('Medical AI Assistant', {timeout:2_000})
  })
})
