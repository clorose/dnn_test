import { Link } from "react-router-dom";

const Header = () => {
  return (
    <header className="fixed top-0 left-0 w-full bg-green-50 z-50 h-8">
      <h1>
        <Link to="/">LOGO</Link>
      </h1>
    </header>
  );
};

export default Header;
