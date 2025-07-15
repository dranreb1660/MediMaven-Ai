export default function Container({ children }: { children: React.ReactNode }) {
  return (
    <div className="w-full mx-auto px-safe px-4 sm:px-6 lg:px-8 max-w-screen-lg">
      {children}
    </div>
  );
}
