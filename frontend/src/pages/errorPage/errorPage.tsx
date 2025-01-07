// src/pages/errorpage/ErrorPage.tsx

const ErrorPage = () => {
  return (
    <div className="flex flex-col items-center justify-center h-screen bg-gray-50">
      <h1 className="text-4xl font-bold text-gray-800 mb-4">404</h1>
      <p className="text-lg text-gray-600">페이지를 찾을 수 없습니다.</p>
    </div>
  );
};

export default ErrorPage;
