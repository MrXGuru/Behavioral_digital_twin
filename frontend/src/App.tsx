import BehavioralTwinDashboard from './BehavioralTwinDashboard'
import OAuthMockPage from './OAuthMockPage'

export default function App() {
  if (window.location.search.includes('oauth_mock=true')) {
    return <OAuthMockPage />
  }
  
  return <BehavioralTwinDashboard />
}

