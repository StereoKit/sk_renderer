// WebGPU pre-init: request the adapter and device BEFORE the wasm runtime
// starts. main() can never block on the web, so skr_init takes the device
// through its pre-provided path (settings.wgpu_device), wrapped from
// Module.preinitializedWebGPUDevice by emscripten_webgpu_get_device().
//
// Property accesses that cross the minifier boundary (anything the runtime
// or library JS reads back) use quoted names so closure builds keep working.

// Surface JS-side failures through the emrun channel — an uncaught exception
// otherwise kills the frame loop with nothing in the captured output
window.addEventListener('error', function (e) {
	(Module['printErr'] || console.error)('[web] uncaught: ' + e.message + ' @ ' + e.filename + ':' + e.lineno);
});
window.addEventListener('unhandledrejection', function (e) {
	(Module['printErr'] || console.error)('[web] unhandled rejection: ' + e.reason);
});

Module['preRun'] = Module['preRun'] || [];
Module['preRun'].push(function () {
	var log = function (msg) { (Module['print'] || console.log)('[web] ' + msg); };
	if (!navigator.gpu) {
		// Leave the device unset; main() reports the missing device cleanly
		log('WebGPU is not available in this browser (navigator.gpu missing)');
		return;
	}
	addRunDependency('skr-webgpu-device');
	navigator.gpu.requestAdapter().then(function (adapter) {
		if (!adapter) throw new Error('navigator.gpu.requestAdapter returned null');
		var info = adapter.info || {};
		log('WebGPU adapter: ' + [info.vendor, info.architecture, info.device, info.description]
			.filter(function (s) { return s; }).join(' / '));
		log('WebGPU adapter features: ' + (Array.from(adapter.features).join(', ') || '(none)'));
		// Take every feature the adapter offers; the renderer probes what it
		// actually has at runtime (wgpuDeviceHasFeature)
		return adapter.requestDevice({ requiredFeatures: Array.from(adapter.features) });
	}).then(function (device) {
		log('WebGPU device ready');
		Module['preinitializedWebGPUDevice'] = device;
		removeRunDependency('skr-webgpu-device');
	}).catch(function (e) {
		log('WebGPU device request failed: ' + e);
		removeRunDependency('skr-webgpu-device');
	});
});
