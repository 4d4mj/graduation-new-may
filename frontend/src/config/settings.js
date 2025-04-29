const settings = {
  apiUrl: process.env.NEXT_PUBLIC_API_URL,   // browser/client fetches
  apiInternalUrl: process.env.API_INTERNAL_URL ?? process.env.NEXT_PUBLIC_API_URL,
};
export default settings;
