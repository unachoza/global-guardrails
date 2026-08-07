import { useState } from "react";
import GuardrailForm from "./GuardrailForm";
import "./App.css";

function App() {
	const [view, setView] = useState("home");
	const [guardrails, setGuardrails] = useState<string[]>([]);

	const handleSubmit = (guardrail: string) => {
		setGuardrails((current) => [...current, guardrail]);
		setView("results");
	};

	return (
		<section id="center">
			<GuardrailForm onSubmit={handleSubmit} />

			{view === "results" && (
				<ul className="guardrail-list">
					{guardrails.map((guardrail, index) => (
						<li key={`${index}-${guardrail}`}>{guardrail}</li>
					))}
				</ul>
			)}
		</section>
	);
}

export default App;
