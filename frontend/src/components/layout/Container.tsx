export default function Container({ children }: { children: React.ReactNode }) {
  return (
    <div className="w-full min-h-screen relative">
      <div className="absolute inset-0 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800" />
      <div className="relative z-10">
        {children}
      </div>
    </div>
  );
}
