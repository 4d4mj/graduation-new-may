/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  async redirects() {
    return [
      {
        source: '/login',          // what the browser asks for
        destination: '/auth/login',// where you really want it
        permanent: false,          // 307 redirect
      },
      {
        source: '/register',       // what the browser asks for
        destination: '/auth/register',// where you really want it
        permanent: false,          // 307 redirect
      },
      {
        source: '/',
        destination: '/dashboard', // where you really want it
        permanent: false,          // 307 redirect
      }
    ];
  },
};

export default nextConfig;
