
import SiteHeader from '../components/ui/SiteHeader';
import WelcomeCard from '../components/ui/WelcomeCard';
import Container from '../components/layout/Container';

export default function WelcomePage() {
  return (
    <div className="bg-gray-50 dark:bg-gray-900 min-h-screen flex flex-col">
      <SiteHeader variant="welcome" />
      <Container>
        <div className="flex flex-1 items-center justify-center px-4">
          <WelcomeCard />
        </div>
      </Container>
    </div>
  );
}
