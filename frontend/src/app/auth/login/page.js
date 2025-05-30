import { LoginForm } from "./Form";

export default function Page() {
	return (
		<div className="flex min-h-svh w-full items-center justify-center p-6 md:p-10">
			<div className="w-full max-w-sm space-y-12">
				<div className="text-center flex-1 bg-gradient-to-r from-slate-800 via-slate-600 to-slate-700 text-transparent bg-clip-text content-center">
					<h1 className="font-bold text-4xl mb-2">MediPal</h1>
					<p className="text-black">
						Your AI-powered health assistant
					</p>
				</div>
				<LoginForm />
			</div>
		</div>
	);
}
