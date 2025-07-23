describe('MediMaven Chat', () => {
  it('loads the app', () => {
    cy.visit('/')
    cy.contains('MediMaven').should('be.visible')
  })

  it('can send a message', () => {
    cy.visit('/')
    cy.get('textarea').type('What are symptoms of diabetes?')
    cy.get('button[type="submit"]').click()
    cy.contains('Thinking', { timeout: 10000 }).should('be.visible')
  })
})