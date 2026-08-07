import { useState } from "react";
import "./App.css";

function App() {
	const [view, setView] = useState("home");

	return (
		<>
			<section id="center">
				<div className="hero">
					<button type="button" className="counter" onClick={() => setView("home")}>
						Enter
					</button>
				</div>
			</section>
		</>
	);
}

export default App;
