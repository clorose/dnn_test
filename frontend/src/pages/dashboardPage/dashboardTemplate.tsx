// path: frontend/src/pages/dashboardPage/dashboardTemplate.tsx
import { Outlet } from "react-router-dom";
import Header from "../../components/common/Header/Header";
import Footer from "../../components/common/Footer/Footer";

const dashboardTemplate = () => {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <div className="flex-1 bg-gray-50 mt-12"> {/* 빨간 배경색 대신 더 중립적인 회색 배경으로 변경 */}
        <Outlet />
      </div>
      <Footer />
    </div>
  );
};

export default dashboardTemplate;