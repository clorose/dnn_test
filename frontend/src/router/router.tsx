import { createBrowserRouter } from "react-router-dom";
import MainPage from "../pages/mainpage/MainPage";
import ErrorPage from "../pages/errorPage/errorPage";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <MainPage />,
    errorElement: <ErrorPage />,
  },
]);
