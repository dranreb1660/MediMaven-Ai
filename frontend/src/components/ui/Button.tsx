type ButtonProps = {
  label: string;
  variant?: 'primary' | 'secondary';
  onClick?: () => void;
  disabled?: boolean;
};

export default function Button({ label, variant = 'primary', onClick, disabled }: ButtonProps) {
  const base =
    'rounded-full px-6 py-2 font-medium transition focus-visible:outline focus-visible:outline-mm-accentDark disabled:opacity-50';
  const styles =
    variant === 'primary'
      ? 'bg-mm-accent text-white hover:bg-mm-accentDark'
      : 'bg-mm-info text-gray-800 hover:bg-mm-info/80';

  return (
    <button className={`${base} ${styles}`} onClick={onClick} disabled={disabled}>
      {label}
    </button>
  );
}
