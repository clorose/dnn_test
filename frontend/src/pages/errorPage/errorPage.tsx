import { useEffect, useRef, useState } from "react";

const ErrorPage = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [showMessage, setShowMessage] = useState<boolean>(false);
  const [isHovered, setIsHovered] = useState<boolean>(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;

    const resizeCanvas = () => {
      if (!canvas) return;
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    const matrixChars: string[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%&*ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃ404".split(
        ""
      );
    const drops: number[] = Array(Math.floor(canvas.width / 15)).fill(0);

    const matrix = () => {
      ctx.fillStyle = "rgba(0, 0, 0, 0.1)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.font = "15px VT323";

      for (let i = 0; i < drops.length; i++) {
        const text =
          matrixChars[Math.floor(Math.random() * matrixChars.length)];
        ctx.fillStyle =
          text === "4" || text === "0"
            ? "rgba(255, 0, 0, 0.8)"
            : "rgba(0, 255, 65, 0.25)";

        ctx.fillText(text, i * 15, drops[i] * 20);

        if (drops[i] * 20 > canvas.height && Math.random() > 0.98) {
          drops[i] = 0;
        }
        drops[i]++;
      }
    };

    const intervalId = setInterval(matrix, 50);

    setTimeout(() => {
      setShowMessage(true);
    }, 1000);

    return () => {
      clearInterval(intervalId);
      window.removeEventListener("resize", resizeCanvas);
    };
  }, []);

  const handleHover = () => {
    setIsHovered((prev) => !prev);
  };

  return (
    <div className="relative min-h-screen bg-black overflow-hidden">
      <canvas
        ref={canvasRef}
        className="absolute top-0 left-0 w-full h-full"
        aria-hidden="true"
      />

      <div className="relative z-10 flex flex-col items-center justify-center min-h-screen">
        <div
          className="text-center p-4"
          onMouseEnter={handleHover}
          onMouseLeave={handleHover}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              handleHover();
            }
          }}
          aria-label={
            isHovered ? "Follow the white rabbit" : "404 error message"
          }
        >
          {showMessage && (
            <>
              <h1 className="text-8xl font-mono text-green-500 mb-8 animate-pulse">
                404
              </h1>
              <p className="text-green-400 text-2xl font-mono">
                <span
                  className={`inline-block transition-opacity duration-300 ${
                    isHovered ? "opacity-0" : "opacity-100"
                  }`}
                >
                  페이지를 찾을 수 없습니다
                </span>
                <span
                  className={`absolute left-0 right-0 transition-opacity duration-300 ${
                    isHovered ? "opacity-100" : "opacity-0"
                  }`}
                >
                  Follow the white rabbit...
                </span>
              </p>
            </>
          )}
        </div>
      </div>

      <link
        href="https://fonts.googleapis.com/css2?family=VT323&display=swap"
        rel="stylesheet"
      />
    </div>
  );
};

export default ErrorPage;
