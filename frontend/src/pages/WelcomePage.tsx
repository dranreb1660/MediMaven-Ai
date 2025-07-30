import SiteHeader from '../components/ui/SiteHeader';
import WelcomeCard from '../components/ui/WelcomeCard';
import Container from '../components/layout/Container';
import Disclaimer from '../components/ui/Disclaimer';

export default function WelcomePage() {
  return (
    <Container>
      <div className="flex justify-center w-full min-h-screen bg-gray-50 dark:bg-gray-900">
        <div className="relative min-h-screen w-full max-w-2xl mx-auto flex flex-col bg-white dark:bg-gray-800">
          
          <SiteHeader variant="welcome" />
          
          <div className="flex-1 flex flex-col">
            <div className="flex-1 flex items-center justify-center p-4">
              <WelcomeCard />
            </div>
            <Disclaimer />
          </div>
          
        </div>
      </div>
    </Container>
  );
}
