export default function Avatar({ size = 44 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <defs>
        <linearGradient id="avatarBg" x1="0" y1="0" x2="64" y2="64" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#bdeed1" />
          <stop offset="1" stopColor="#8fe0b6" />
        </linearGradient>
        <linearGradient id="avatarHair" x1="10" y1="8" x2="54" y2="34" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#5b4636" />
          <stop offset="1" stopColor="#3f2f24" />
        </linearGradient>
      </defs>

      {/* background */}
      <circle cx="32" cy="32" r="30" fill="url(#avatarBg)" />

      {/* neck + shoulders */}
      <path d="M20 58c1-8 5-12 12-12s11 4 12 12" fill="#ffffff" fillOpacity="0.9" />

      {/* hair back */}
      <path
        d="M14 30c0-11 8-19 18-19s18 8 18 19c0 3-1 6-2 8-1-6-4-9-6-9-3 5-9 6-16 6-4 0-7-2-9-5-2 2-3 4-3 8-1-3-2-5-2-8z"
        fill="url(#avatarHair)"
      />

      {/* face */}
      <circle cx="32" cy="29" r="14" fill="#f3c9a3" />

      {/* ears */}
      <circle cx="18.5" cy="30" r="2.6" fill="#f3c9a3" />
      <circle cx="45.5" cy="30" r="2.6" fill="#f3c9a3" />

      {/* hair front / fringe */}
      <path
        d="M18 24c1-7 6-12 14-12s13 5 14 12c-3-2-7-3-14-3s-11 1-14 3z"
        fill="url(#avatarHair)"
      />

      {/* blush */}
      <circle cx="24" cy="33" r="2.4" fill="#f0a37a" fillOpacity="0.55" />
      <circle cx="40" cy="33" r="2.4" fill="#f0a37a" fillOpacity="0.55" />

      {/* eyes */}
      <path d="M23 28.5c1.4-1.4 4-1.4 5.2 0" stroke="#3f2f24" strokeWidth="1.8" strokeLinecap="round" fill="none" />
      <path d="M35.8 28.5c1.4-1.4 4-1.4 5.2 0" stroke="#3f2f24" strokeWidth="1.8" strokeLinecap="round" fill="none" />

      {/* smile */}
      <path d="M25 35.5c2.6 2.6 9.4 2.6 12 0" stroke="#8a4b33" strokeWidth="2" strokeLinecap="round" fill="none" />

      {/* headset - signals "here to help" */}
      <path
        d="M16 27a16 16 0 0 1 32 0"
        stroke="#23945c"
        strokeWidth="2.4"
        strokeLinecap="round"
        fill="none"
      />
      <rect x="13.4" y="26" width="4.4" height="7" rx="2.2" fill="#23945c" />
      <rect x="46.2" y="26" width="4.4" height="7" rx="2.2" fill="#23945c" />
      <path d="M48.4 32.5v3a4 4 0 0 1-4 4h-2.8" stroke="#23945c" strokeWidth="2.2" strokeLinecap="round" fill="none" />
      <circle cx="41.2" cy="39.5" r="1.8" fill="#23945c" />
    </svg>
  );
}
