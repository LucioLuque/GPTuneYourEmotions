const clientId = 'bdbf0e0e3bf24066aa88ad22c5841cf7';
// const redirectUri = 'http://127.0.0.1:5500/GPTuneYourEmotions.html';
// const redirectUri = 'http://127.0.0.1:5000/';
// const redirectUri = 'http://127.0.0.1:5500/GPTuneYourEmotions.html';
const redirectUri = 'http://127.0.0.1:5500/web/GPTuneYourEmotions.html';

const generateRandomString = (length) => {
  const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  const values = crypto.getRandomValues(new Uint8Array(length));
  return values.reduce((acc, x) => acc + possible[x % possible.length], "");
};

const sha256 = async (plain) => {
  const encoder = new TextEncoder();
  const data = encoder.encode(plain);
  return window.crypto.subtle.digest('SHA-256', data);
};

const base64encode = (input) => {
  return btoa(String.fromCharCode(...new Uint8Array(input)))
    .replace(/=/g, '')
    .replace(/\+/g, '-')
    .replace(/\//g, '_');
};

export async function login() {
  const urlParams = new URLSearchParams(window.location.search);
  const code = urlParams.get('code');

  if (!code) {
    const codeVerifier = generateRandomString(64);
    const hashed = await sha256(codeVerifier);
    const codeChallenge = base64encode(hashed);

    localStorage.setItem('code_verifier', codeVerifier);

    // const scope = 'user-read-private user-read-email user-top-read';
    const scope = [
    'user-read-private',
    'user-read-email',
    'user-top-read',
    'playlist-modify-private',   // para poder crear/editar playlists privadas
    'playlist-modify-public'     // para poder crear/editar playlists públicas
    ].join(' ');
    const authUrl = new URL("https://accounts.spotify.com/authorize");
    authUrl.search = new URLSearchParams({
      response_type: 'code',
      client_id: clientId,
      scope,
      code_challenge_method: 'S256',
      code_challenge: codeChallenge,
      redirect_uri: redirectUri,
    }).toString();

    window.location.href = authUrl.toString();
    return [];
  }

  const codeVerifier = localStorage.getItem('code_verifier');

  const tokenRes = await fetch("https://accounts.spotify.com/api/token", {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      client_id: clientId,
      grant_type: 'authorization_code',
      code,
      redirect_uri: redirectUri,
      code_verifier: codeVerifier,
    }),
  });

  if (!tokenRes.ok) {
    const errText = await tokenRes.text();
    console.error('💥 /api/token error:', tokenRes.status, errText);
    throw new Error(`Token exchange failed: ${tokenRes.status}`);
  }
  const tokenData = await tokenRes.json();
  const accessToken = tokenData.access_token;
  if (!accessToken) {
    console.error('🚫 No recibí access_token:', tokenData);
    throw new Error('No se obtuvo access_token del servidor de Spotify.');
  }

  const userProfileRes = await fetch("https://api.spotify.com/v1/me", {
    headers: { Authorization: `Bearer ${accessToken}` }
  });

  if (!userProfileRes.ok) {
    const errText = await userProfileRes.text();
    console.error('💥 /v1/me error:', userProfileRes.status, errText);
    throw new Error(`Spotify /me failed: ${userProfileRes.status}`);
  }
  const userProfile = await userProfileRes.json();
  const profileImg = userProfile.images?.[0]?.url;
  if (profileImg) {
    localStorage.setItem('spotify_profile_img', profileImg);
  }

  const artistRes = await fetch("https://api.spotify.com/v1/me/top/artists?limit=50&time_range=long_term", {
    headers: { Authorization: `Bearer ${accessToken}` }
  });

  const artistData = await artistRes.json();
  const genres = artistData.items.flatMap(artist => artist.genres);

  const genreCounts = {};
  genres.forEach(genre => {
    genreCounts[genre] = (genreCounts[genre] || 0) + 1;
  });

  const sortedGenres = Object.entries(genreCounts)
    .sort((a, b) => b[1] - a[1])
    .map(([genre, count]) => `${genre}: ${count}`);

  if (window.history.pushState) {
    const cleanUrl = window.location.origin + window.location.pathname;
    window.history.pushState({}, document.title, cleanUrl);
  }
  localStorage.setItem('spotify_token', accessToken);
  return sortedGenres;
}
