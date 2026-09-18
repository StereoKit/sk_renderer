// WebGPU + WebXR pre-init for the XR example.
//
// This is example/web/pre.js with one difference that matters enormously:
// the adapter is requested with {xrCompatible: true}. Without it, WebOpenXR's
// xrCreateSession fails at `new XRGPUBinding(session, device)` with
// "InvalidStateError", and there is no earlier warning - the adapter, the
// device and every ordinary draw all work perfectly first.
//
// The device has to exist before the wasm runtime starts, because main() can
// never block on the web; skr_init takes it through Module.preinitializedWebGPUDevice.
//
// Property accesses that cross the minifier boundary use quoted names so
// closure builds keep working.

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
		log('WebGPU is not available in this browser (navigator.gpu missing)');
		return;
	}
	addRunDependency('skr-webgpu-device');
	navigator.gpu.requestAdapter({ xrCompatible: true }).then(function (adapter) {
		if (!adapter) throw new Error('navigator.gpu.requestAdapter returned null');
		var info = adapter.info || {};
		log('WebGPU adapter: ' + [info.vendor, info.architecture, info.device, info.description]
			.filter(function (s) { return s; }).join(' / '));
		// Take every feature the adapter offers; the renderer probes what it
		// actually has at runtime (wgpuDeviceHasFeature)
		return adapter.requestDevice({ requiredFeatures: Array.from(adapter.features) });
	}).then(function (device) {
		log('WebGPU device ready');
		// Dawn reports validation failures to the console. In a headset there is
		// no console to read, so a rejected render pass is indistinguishable from
		// a black screen with a healthy log. Route them somewhere visible.
		device.addEventListener('uncapturederror', function (e) {
			(Module['printErr'] || console.error)('[wgpu] ' + e.error.constructor.name + ': ' + e.error.message);
		});
		device.lost.then(function (info) {
			(Module['printErr'] || console.error)('[wgpu] device lost: ' + info.reason + ' ' + info.message);
		});
		// The two facts that decide whether this run can enter VR at all.
		log('WebXR: navigator.xr=' + (!!navigator.xr) +
		    ', XRGPUBinding=' + (typeof XRGPUBinding !== 'undefined'));
		Module['preinitializedWebGPUDevice'] = device;
		removeRunDependency('skr-webgpu-device');
	}).catch(function (e) {
		log('WebGPU device request failed: ' + e);
		removeRunDependency('skr-webgpu-device');
	});
});
