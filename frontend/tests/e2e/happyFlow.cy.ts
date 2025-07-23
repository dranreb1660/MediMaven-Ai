describe('Happy path', () => {
  it('welcome → two queries → new chat', () => {
    cy.visit('/')
    cy.get('input[placeholder*="Describe"]').type('What is diabetes?{enter}')
    cy.url().should('include','/chat')
    cy.contains('diabetes', {timeout:12_000}).should('be.visible')

    cy.get('textarea').type('is it curable?{enter}')
    cy.contains('curable').should('be.visible')

    cy.contains('Clear chat').click()
    cy.contains('medical assistant', {timeout:2_000})
  })
})
