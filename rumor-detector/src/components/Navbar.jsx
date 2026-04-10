import { Link, useLocation } from 'react-router-dom'

const Navbar = () => {
  const location = useLocation()

  return (
    <nav className="navbar">
      <div className="nav-container">
        <Link to="/" className="nav-logo">
          <span className="logo-icon">🔍</span>
          <span className="logo-text">TruthGuard</span>
        </Link>

        <div className="nav-links">
          <Link to="/" className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}>
            Home
          </Link>
          <Link to="/how-we-analyze" className={`nav-link ${location.pathname === '/how-we-analyze' ? 'active' : ''}`}>
            How we Analyze
          </Link>
          <Link to="/verify" className={`nav-link ${location.pathname === '/verify' ? 'active' : ''}`}>
            Verify News
          </Link>
        </div>
      </div>
    </nav>
  )
}

export default Navbar
